from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import cv2
import mujoco as mj
import trimesh
from TaximSensor.Basics.CalibData import CalibData, read_calib_np
import TaximSensor.Basics.params as pr
import TaximSensor.Basics.sensorParams as psp

__version__ = "0.1"  # Source of truth for mujoco-taxim's version

_exported_dunders = {
    "__version__",
}

def invert_homogeneous_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv

@dataclass
class Link:
    """
    Dataset class for objects in MuJoCo.
    """
    obj_id: int  # MuJoCo object ID
    obj_type: mj.mjtObj  # MuJoCo object type
    mujoco_data: any = None
    mujoco_model: any = None
    obj_name: str = None

    # get pose from mujoco
    def get_pose(self):
        """
        Gets the pose of the object in world coordinates, with x-axis flipped for pyrender.
        """
        if self.obj_type == mj.mjtObj.mjOBJ_SITE:
            # Camera is created from a site, so we need to access a different data
            # Get the world-space position and orientation (rotation matrix)
            position = self.mujoco_data.site_xpos[self.obj_id].copy()
            orientation = self.mujoco_data.site_xmat[self.obj_id].reshape(3, 3).copy()

        # Pyrender camera has a RHS convention, but geoms use LHS; this makes it 90 deg off about x-axis
        elif self.obj_type == mj.mjtObj.mjOBJ_BODY:
            # For bodies, just xpos / xmat is fine
            position = self.mujoco_data.xpos[self.obj_id].copy()
            orientation = self.mujoco_data.xmat[self.obj_id].reshape(3, 3).copy()

        elif self.obj_type == mj.mjtObj.mjOBJ_GEOM:
            # For geom, fetch from geom_*
            position = self.mujoco_data.geom_xpos[self.obj_id].copy()
            orientation = self.mujoco_data.geom_xmat[self.obj_id].reshape(3, 3).copy()

        else:
            # Handle other object types if needed
            raise NotImplementedError(
                f"Object type {self.obj_type} not implemented for pose retrieval.")
        return position, orientation


class TaximSensor(object):
    def __init__(self, sensor_type="digit", bg_file=None, bg_index=0):
        '''
        Initialize the simulator.
        1) load the calibration files,
        2) generate shadow table from shadow masks
        3) load the gelpad model

        :param self: Description
        :param data_folder: root path to calibration data
        :param gelpad_model_path: path to the gelpad model numpy file
        ''' 
        if sensor_type != "digit":
            raise NotImplementedError("Currently only digit sensor is supported.")

        self.sensor_type = sensor_type
        self.obj_pointclouds = {}
        self.obj_mesh = {}
        self.object_links = {}
        self.object_body_ids = set()
        self.saved=False 
        # polytable
        calib_data = f"{sensor_type}/polycalib.npz"
        self.calib_data = CalibData(calib_data)

        # raw calibration data, here only used for background
        if bg_file is None:
            data_file = read_calib_np(f"{sensor_type}/bg_set.npz")
        else:
            data_file = np.load(bg_file, allow_pickle=True)
        self.data_file = data_file['f0']
        self.bgs = []
        for i in range(self.data_file.shape[0]):
            self.f0 = self.data_file[i]
            self.bgs.append(self.processInitialFrame())
        
        self.f0 = self.data_file[bg_index]
        self.bg_proc = self.bgs[bg_index]
        self.bg_len = len(self.bgs)
        self.bg_index = bg_index
        self.bg_proc = self.bgs[bg_index]

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = read_calib_np("shadowTable.npz")
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

        self.gel_map = read_calib_np("gelmap5.npy")
        self.gel_map = cv2.GaussianBlur(self.gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

    def change_bg(self, bg_index):
        if bg_index > len(self.bgs)-1:
            print("Warning: bg_index exceeds the number of available backgrounds. No change made.")
            return
        self.f0 = self.data_file[bg_index]
        self.bg_proc = self.bgs[bg_index]
        self.bg_index = bg_index

    def add_object_mujoco(self, obj_name, model, data, mesh_name=None, obj_type=mj.mjtObj.mjOBJ_BODY):
        """
        Add an object to the list of objects to be tracked by the sensor.
        The given obj_name is used to find the corresponding mesh's name as defined in the xml, by appending _mesh.
        e.g. if obj_name is "box_geom", the mesh name must be "box_geom_mesh".
        This mesh is converted to pointcloud format and tracked by the sensor in subsequent updates.
        Since it requires the corresponding object body's pose in the simulation, at least one mj_step should be called
        before this function.

        :param obj_name: str
            Name of the body to be added. This is defined as a mujoco body, and its associated mesh is expected to be
            defined in the mujoco model with the name obj_name + "_mesh", unless provided otherwise.
        :param model: mjModel
        :param data: mjData
        :param mesh_name: str, optional
            Name of the mesh to be used for the object. If not provided, it defaults to obj_name + "_mesh".
            This is useful if the mesh name differs from the default convention of appending "_mesh" to the body name.
        :param obj_type: mj.mjtObj, optional
            either a mjOBJ_BODY or mjOBJ_GEOM. Defaults to mjOBJ_BODY.
        """
        if(obj_type == mj.mjtObj.mjOBJ_BODY):
            obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
            body_id = obj_id
        elif(obj_type == mj.mjtObj.mjOBJ_GEOM):
            obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, obj_name)
            body_id = model.geom_bodyid[obj_id]
        else:
            raise ValueError(f"Unsupported object type: {obj_type}")

        # Keep track of body id for contact checking
        self.object_body_ids.add(body_id)
        self.object_links[obj_name] = Link(
            obj_id, obj_type, data, model, obj_name
        )
        # position, orientation = self.object_links[obj_name].get_pose()

        if(obj_type == mj.mjtObj.mjOBJ_GEOM):
            # if obj_type=GEOM, we need to check if it is a mesh or a primitive
            geom_type = model.geom_type[obj_id]

            if(geom_type == mj.mjtGeom.mjGEOM_MESH):
                # if mesh, use the mesh name for creating the trimesh
                # Construct the trimesh
                mesh_name = obj_name + "_mesh" if mesh_name is None else mesh_name
                mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
                assert mesh_id >= 0, f"Mesh {mesh_name} not found in model."
                obj_pc, obj_mesh = self.build_pointcloud_from_mujoco_mesh(model, mesh_id)
            else:
                obj_pc, obj_mesh = self.build_pointcloud_from_mujoco_primitive(model, obj_id, geom_type)
        else: 
            # if obj_type=BODY, we assume it has a corresponding mesh defined in the model
            # Construct the trimesh
            mesh_name = obj_name + "_mesh" if mesh_name is None else mesh_name
            mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
            assert mesh_id >= 0, f"Mesh {mesh_name} not found in model."
            obj_pc, obj_mesh = self.build_pointcloud_from_mujoco_mesh(model, mesh_id)
        self.obj_pointclouds[obj_name] = obj_pc * 1000
        self.obj_mesh[obj_name] = obj_mesh

    def add_body_mujoco(self, body, model, data, mesh_name=None):
        '''
        Convenience function that wraps add_object_mujoco for mjOBJ_BODY type objects.
        '''
        self.add_object_mujoco(body, model, data, mesh_name=mesh_name, obj_type=mj.mjtObj.mjOBJ_BODY)

    def add_geom_mujoco(self, geom, model, data, mesh_name=None):
        '''
        Convenience function that wraps add_object_mujoco for mjOBJ_GEOM type objects.
        '''
        self.add_object_mujoco(geom, model, data, mesh_name=mesh_name, obj_type=mj.mjtObj.mjOBJ_GEOM)

    def add_camera_mujoco(self, sensor_name, model, data):
        """
        Queries the MuJoCo model for the site corresponding to the given sensor_name.
        The site is associated with the plane that Taxim will render from.
        In addition, we store the associated site's body_id in sensor_body_ids for later use.
        :param sensor_name: str
            Name of the sensor to be added. This is defined as a mujoco.sensor.touch_Grid plugin, and its name
              should match the name of its associated site in the mujoco model.
        :param model: mjModel
        :param data: mjData
        """
        # Get the site ID using its name
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, sensor_name)

        # Create the camera to be passed to pyrender
        self.sensor = Link(
            site_id, mj.mjtObj.mjOBJ_SITE, data, model, sensor_name
        )
        # Keep track of the number of cameras
        # Remember what the associated site's body_id is for contact checking
        self.sensor_body_id = model.site_bodyid[site_id]
        self.sensor_name = sensor_name

    def build_pointcloud_from_mujoco_mesh(self, model, mesh_id, n_points=9999999, seed=None):
        """
        Sample a point cloud (Nx3 numpy array) from a MuJoCo mesh using trimesh utilities.

        Parameters
        ----------
        model : mjModel
        mesh_id : int
        n_points : int
            Number of points to sample on the surface.
        seed : int | None
            Random seed for deterministic sampling.

        Returns
        -------
        np.ndarray
            (n_points, 3) float array of sampled points.
        """
        # --- build trimesh from MuJoCo mesh buffers (same as before) ---
        start_vert = model.mesh_vertadr[mesh_id]
        num_vert = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start_vert : start_vert + num_vert].reshape(-1, 3)

        start_face = model.mesh_faceadr[mesh_id]
        num_face = model.mesh_facenum[mesh_id]
        faces = model.mesh_face[start_face : start_face + num_face].reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # --- trimesh-built-in surface sampling ---
        if seed is not None:
            rng = np.random.default_rng(seed)
            points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng)
        else:
            points, _ = trimesh.sample.sample_surface(mesh, n_points)
        
        return np.asarray(points, dtype=np.float32), mesh


    def build_pointcloud_from_mujoco_primitive(self, model, geom_id, geom_type, n_points=999999, seed=None):
        """
        Sample a point cloud (Nx3 numpy array) from a MuJoCo primitive geom by creating the
        corresponding trimesh primitive and sampling its surface.

        Parameters
        ----------
        model : mjModel
        geom_id : int
        geom_type : int
            model.geom_type[geom_id]
        n_points : int
            Number of points to sample on the surface.
        seed : int | None
            Random seed for deterministic sampling.

        Returns
        -------
        np.ndarray
            (n_points, 3) float array of sampled points.
        """
        geom_type_map = {
            2: "sphere",     # mjGEOM_SPHERE
            3: "capsule",    # mjGEOM_CAPSULE
            4: "ellipsoid",  # mjGEOM_ELLIPSOID
            5: "cylinder",   # mjGEOM_CYLINDER
            6: "box",        # mjGEOM_BOX
        }

        kind = geom_type_map.get(geom_type, None)
        size = model.geom_size[geom_id]

        if kind == "sphere":
            radius = float(size[0])
            mesh = trimesh.creation.icosphere(radius=radius)

        elif kind == "cylinder":
            radius = float(size[0])
            height = float(2.0 * size[1])  # MuJoCo uses half-length
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

        elif kind == "box":
            extents = (2.0 * size[:3]).astype(float)  # MuJoCo uses half-extents
            mesh = trimesh.creation.box(extents=extents)

        elif kind == "capsule":
            radius = float(size[0])
            height = float(2.0 * size[1])  # MuJoCo uses half-length (cyl part)
            mesh = trimesh.creation.capsule(radius=radius, height=height, count=[32, 16])

        elif kind == "ellipsoid":
            # Approximate ellipsoid by scaling a unit sphere
            mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
            mesh.apply_scale(size[:3])

        else:
            raise NotImplementedError(
                f"Primitive geom_type '{kind}' not supported or unknown (type id: {geom_type})"
            )

        # --- trimesh-built-in surface sampling ---
        if seed is not None:
            rng = np.random.default_rng(seed)
            points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng)
        else:
            points, _ = trimesh.sample.sample_surface(mesh, n_points)

        return np.asarray(points, dtype=np.float32), mesh


    def get_force_mujoco(self, model, data):
        """
        Runs a contact check between the sensor and the objects in the scene.
        If a contact between the sensor and an object of interest is found,
        it fetches the touch grid data from the mujoco sensor and returns it.
        Else, it returns None, to prevent unnecessary rendering of the sensor.

        """
        # We want the key to the dict to be either a body name or a geom name,
        # depending on what was added
        sensor_body_id = self.sensor_body_id
        b1 = None
        b2 = None
        b1_name = None
        b2_name = None
        got_contact = False
        if len(data.contact) == 0:
            return None
        for c in data.contact:
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            b1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            b2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)

            g1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1)
            g2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2)

            if (b1 == sensor_body_id or b1 in self.object_body_ids) and (
                b2 == sensor_body_id or b2 in self.object_body_ids
            ):
                # If the contact is between tacto body and object body, we are interested in the force data
                got_contact = True
                break
        if not got_contact:
            return None

        # Fetch touch grid data
        sensor_id = model.sensor(self.sensor_name).id
        touch_data = data.sensordata[
            sensor_id : sensor_id + model.sensor_dim[sensor_id]
        ].reshape((120, 160, 3))
        touch_data = touch_data[:, :, 0]  # get only the normal forces

        # get the object names in contact with the sensor
        if b1 == sensor_body_id:
            obj_name = b2_name if b2_name in self.object_links.keys() else g2_name
        else: # b2 == sensor_body_id
            obj_name = b1_name if b1_name in self.object_links.keys() else g1_name
        # obj_name = b1_name if b2 == sensor_body_id else b2_name
        touch_data = {obj_name: touch_data}
        # TODO: Make the dict key distinct for different sensors
        return touch_data

    def render_taxim_named(self, model, data, name, shadow=True, get_depth=True, visualize=True):
        '''
        Renders the taxim image based on the current mujoco state.
        1. Check for contact with self.get_force_mujoco
        2. Fetch the wTs and wTo
        3. Pass it to the simulator to generate the tactile image
        4. Return the image
        
        :param self: Description
        :param model: Description
        :param data: Description
        '''
        
        obj_name = name
        wPs, wRs = self.sensor.get_pose()
        wTs = np.eye(4)
        wTs[:3, :3] = wRs
        wTs[:3, 3] = wPs #* 1000.0 # change to mm
        wPo, wRo = self.object_links[obj_name].get_pose()
        wTo = np.eye(4)
        wTo[:3, :3] = wRo
        wTo[:3, 3] = wPo #* 1000.0 # change to mm

        height_map, gel_map, contact_mask, press_depth, gt_height_map = self.generateHeightMapWithTransform(wTs, wTo, obj_name)
        heightMap, contact_mask, contact_height = self.deformApprox(press_depth, height_map, gel_map, contact_mask)
        sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
        sim_img = sim_img if not shadow else shadow_sim_img
        
        # add some gaussian noise to simulate real sensor noise
        noise_sigma = 5
        noise = np.random.normal(0, noise_sigma, sim_img.shape).astype(sim_img.dtype)
        sim_img = cv2.add(sim_img, noise)
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if(visualize):
            if not get_depth:
                combined_img = sim_img
            else:
                gt_height_map  = cv2.rotate(gt_height_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # repeat height map to 3 channels
                gt_vis = np.repeat(gt_height_map[:, :, np.newaxis], 3, axis=2)
                div = 1 if np.max(gt_vis) == 0 else np.max(gt_vis)
                gt_vis = (gt_vis / div * 255).astype(np.uint8)
                combined_img = np.concatenate((sim_img, gt_vis), axis=1)
            cv2.imshow("taxim", combined_img)
            cv2.waitKey(1)
        if get_depth:
            return sim_img, gt_height_map
        else:
            return sim_img, np.zeros((psp.w, psp.h))
        
    def render_taxim(self, model, data, shadow=True, get_depth=True, visualize=True):
        '''
        Renders the taxim image based on the current mujoco state.
        1. Check for contact with self.get_force_mujoco
        2. Fetch the wTs and wTo
        3. Pass it to the simulator to generate the tactile image
        4. Return the image
        
        :param self: Description
        :param model: Description
        :param data: Description
        '''
        touch_data = self.get_force_mujoco(model, data)
        if touch_data is None:
            sim_img = self.bg_proc.astype(np.float64)
            gt_height_map = np.zeros((psp.h, psp.w))
        else:
            obj_name = [*touch_data][0]
            wPs, wRs = self.sensor.get_pose()
            wTs = np.eye(4)
            wTs[:3, :3] = wRs
            wTs[:3, 3] = wPs * 1000.0 # change to mm
            wPo, wRo = self.object_links[obj_name].get_pose()
            wTo = np.eye(4)
            wTo[:3, :3] = wRo
            wTo[:3, 3] = wPo * 1000.0 # change to mm

            height_map, gel_map, contact_mask, press_depth, gt_height_map = self.generateHeightMapWithTransform(wTs, wTo, obj_name)
            heightMap, contact_mask, contact_height = self.deformApprox(press_depth, height_map, gel_map, contact_mask)
            sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
            sim_img = sim_img if not shadow else shadow_sim_img
        
        # add some gaussian noise to simulate real sensor noise
        noise_sigma = 5
        noise = np.random.normal(0, noise_sigma, sim_img.shape).astype(sim_img.dtype)
        sim_img = cv2.add(sim_img, noise)
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if(visualize):
            if not get_depth:
                combined_img = sim_img
            else:
                gt_height_map  = cv2.rotate(gt_height_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # repeat height map to 3 channels
                gt_vis = np.repeat(gt_height_map[:, :, np.newaxis], 3, axis=2)
                div = 1 if np.max(gt_vis) == 0 else np.max(gt_vis)
                gt_vis = (gt_vis / div * 255).astype(np.uint8)
                combined_img = np.concatenate((sim_img, gt_vis), axis=1)
            cv2.imshow("taxim", combined_img)
            cv2.waitKey(1)
        if get_depth:
            return sim_img, gt_height_map
        else:
            return sim_img, np.zeros((psp.w, psp.h))
        
    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]
        return f0

    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)
        

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        if(height_idx.size == 0):
            return sim_img, sim_img
        
        height_idx_max = int(np.max(height_idx))
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            frame_back = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img

    def rasterize_depth_from_trimesh(
        self,
        mesh: trimesh.Trimesh,
        sTo: np.ndarray,
        H: int,
        W: int,
        pixmm: float,
    ) -> np.ndarray:
        """
        Returns zbuf (H,W) storing minimum z in *sensor frame* for each pixel.
        z is in the same units as mesh coordinates (typically meters or mm).
        """
        # --- Transform mesh vertices into sensor frame ---

        V = (mesh.vertices.astype(np.float32) * 1000)      # (V,3)
        F = mesh.faces.astype(np.int32)                    # (F,3)

        Vh = np.c_[V, np.ones((len(V), 1), dtype=np.float32)]
        Vs = (sTo @ Vh.T).T[:, :3]                         # (V,3) in sensor frame

        # Gather triangles in sensor frame: (F,3,3)
        tris = Vs[F]

        # --- Project triangle vertices to pixel coords (float) ---
        us = tris[..., 0] / pixmm + (W * 0.5)
        vs = tris[..., 1] / pixmm + (H * 0.5)
        zs = tris[..., 2].astype(np.float32)

        zbuf = np.full((H, W), np.inf, dtype=np.float32)

        # --- Rasterize each triangle into zbuf ---
        for f in range(tris.shape[0]):
            u0, u1, u2 = us[f]
            v0, v1, v2 = vs[f]
            z0, z1, z2 = zs[f]

            # Pixel-space bounding box
            umin = int(np.floor(min(u0, u1, u2)))
            umax = int(np.ceil (max(u0, u1, u2)))
            vmin = int(np.floor(min(v0, v1, v2)))
            vmax = int(np.ceil (max(v0, v1, v2)))

            # Quickly skip if outside image
            if umax < 0 or umin >= W or vmax < 0 or vmin >= H:
                continue

            # Clip bbox to image
            umin = max(umin, 0); umax = min(umax, W - 1)
            vmin = max(vmin, 0); vmax = min(vmax, H - 1)

            # Degenerate triangle?
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if denom == 0.0:
                continue
            inv_denom = 1.0 / denom

            # Rasterize bbox
            for v in range(vmin, vmax + 1):
                py = v + 0.5
                for u in range(umin, umax + 1):
                    px = u + 0.5

                    w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv_denom
                    w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv_denom
                    w2 = 1.0 - w0 - w1

                    # Inside test (top-left rule not implemented; this is usually fine for sensor sim)
                    if (w0 >= 0.0) and (w1 >= 0.0) and (w2 >= 0.0):
                        z = w0 * z0 + w1 * z1 + w2 * z2
                        if z < zbuf[v, u]:
                            zbuf[v, u] = z
        # breakpoint()
        return zbuf
    
    def heightmap_from_zbuf(self, zbuf: np.ndarray, pixmm: float) -> np.ndarray:
        """
        Matches your convention: keep only points below gel surface (z < 0),
        and height = -z / pixmm.
        """
        height = np.zeros_like(zbuf, dtype=np.float32)
        hit = np.isfinite(zbuf) & (zbuf < 0.0)
        height[hit] = -zbuf[hit] / pixmm
        return height

    def generateHeightMapWithTransform(self, wTs, wTo, obj_name):
        """
        Generate the height map by interacting the object with the gelpad model.

        wTs: world to sensor transformation matrix
        wTo: world to object transformation matrix
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # assert(self.obj_pointclouds[obj_name].shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = self.gel_map.copy()
        heightMap = np.zeros((psp.h,psp.w))

        # calculate sTo: object-in-sensor-frame transform
        sTw = invert_homogeneous_matrix(wTs)
        sTo = sTw @ wTo

        # Rasterize the depth of the object in sensor frame
        zbuf = self.rasterize_depth_from_trimesh(
            self.obj_mesh[obj_name],
            sTo,
            psp.h,
            psp.w,
            psp.pixmm,
        )
        heightMap = self.heightmap_from_zbuf(zbuf, psp.pixmm)
        # pressing depth in pixel
        valid = np.isfinite(zbuf)          # pixels where mesh projects
        if np.any(valid):
            min_z = np.min(zbuf[valid])    # most negative z (deepest), or could be >0 if mesh is above gel
            pressing_height_mm = min(3.0, max(0.0, -min_z))
        else:
            pressing_height_mm = 0.0
            
        # # Original approach
        # # Convert vertices (sampled pc) to sensor frame
        # wVertices_h = np.hstack((self.obj_pointclouds[obj_name].copy(), np.ones((self.obj_pointclouds[obj_name].shape[0], 1))))
        # sVertices_h = (sTo @ wVertices_h.T).T

        # # Change xy to sensor pixel coordinate
        # uu = (sVertices_h[:,0]/psp.pixmm + psp.w//2).astype(int)
        # vv = (sVertices_h[:,1]/psp.pixmm + psp.h//2).astype(int)
        # # Check boundary of the image 
        # mask_u = np.logical_and(uu > 0, uu < psp.w)
        # mask_v = np.logical_and(vv > 0, vv < psp.h)
        # # Check the depth, only keep points that are below the gelpad surface
        # mask_z = sVertices_h[:,2] < 0 # 0 in gelpad coordinates.
        # mask_map = mask_u & mask_v & mask_z

        # # Get the minimum z per pixel to avoid artifacts
        # u = uu[mask_map]
        # v = vv[mask_map]
        # z = sVertices_h[mask_map, 2]           # this is what you want to minimize per pixel

        # # 1) Build a depth buffer that stores the minimum z per pixel
        # zbuf = np.full((psp.h, psp.w), np.inf, dtype=np.float32)
        # np.minimum.at(zbuf, (v, u), z)     # for each (v,u), keep min(z)

        # # 2) Convert to heightMap (only where something hit)
        # heightMap = np.zeros((psp.h, psp.w), dtype=np.float32)  # or whatever default you want
        # hit = np.isfinite(zbuf)
        # heightMap[hit] = -zbuf[hit] / psp.pixmm
        # # pressing depth in pixel
        # try:
        #     pressing_height_mm = min(3, -np.min(sVertices_h[:,2]))
        # except:
        #     pressing_height_mm = 0
        
        pressing_height_pix = pressing_height_mm/psp.pixmm
        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)

        # get the contact area 
        contact_mask = heightMap > gel_map # heightMap > gel_map
        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]
        heightMapBlur = cv2.GaussianBlur(heightMap.astype(np.float32)/heightMap.max(),(5,5),0)
        return zq, gel_map, contact_mask, pressing_height_mm, heightMapBlur

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        contact_height = zq - gel_map
        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        
        return GD1

    def generate_normals(self,height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')
