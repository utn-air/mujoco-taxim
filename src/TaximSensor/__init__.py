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
import TaximSensor.Core as Core
from TaximSensor.helpers import smooth_mesh, invert_homogeneous_matrix, smooth_heightmap_mm, build_trimesh_from_mujoco_mesh, build_trimesh_from_mujoco_primitive
__version__ = "0.1"  # Source of truth for mujoco-taxim's version

_exported_dunders = {
    "__version__",
}

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
    def __init__(self, sensor_type="digit", bg_file=None, bg_index=0, resize=None):
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
        self.obj_mesh = {}
        self.object_links = {}
        self.object_body_ids = set()
        self.saved=False 
        # polytable
        calib_data = f"{sensor_type}/polycalib.npz"
        self.calib_data = CalibData(calib_data)
        self.resize=resize

        # raw calibration data, here only used for background
        if bg_file is None:
            data_file = read_calib_np(f"{sensor_type}/bg_set.npz")
        else:
            data_file = np.load(bg_file, allow_pickle=True)
        self.data_file = data_file['f0']
        self.bgs = []
        self.bgs_rot = []
        for i in range(self.data_file.shape[0]):
            self.f0 = self.data_file[i]
            self.bgs.append(self.processInitialFrame())
            self.bgs_rot.append(cv2.rotate(np.clip(np.rint(self.bgs[-1].copy()), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE))
            if self.resize is not None:
                self.bgs_rot[-1] = cv2.resize(self.bgs_rot[-1], self.resize)
        
        self.f0 = self.data_file[bg_index]
        self.bg_len = len(self.bgs)
        self.bg_index = bg_index
        self.bg_proc = self.bgs[bg_index]
        self.bg_proc_rot = self.bgs_rot[bg_index]

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
        self.bg_proc_rot = self.bgs_rot[bg_index]
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
        assert obj_id >= 0, f"Object {obj_name} not found in model."
        # Keep track of body id for contact checking
        self.object_body_ids.add(body_id)
        self.object_links[obj_name] = Link(
            obj_id, obj_type, data, model, obj_name
        )

        if(obj_type == mj.mjtObj.mjOBJ_GEOM):
            # if obj_type=GEOM, we need to check if it is a mesh or a primitive
            geom_type = model.geom_type[obj_id]

            if(geom_type == mj.mjtGeom.mjGEOM_MESH):
                # if mesh, use the mesh name for creating the trimesh
                # Construct the trimesh
                mesh_name = obj_name + "_mesh" if mesh_name is None else mesh_name
                mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
                assert mesh_id >= 0, f"Mesh {mesh_name} not found in model."
                obj_mesh = build_trimesh_from_mujoco_mesh(model, mesh_id)
            else:
                obj_mesh = build_trimesh_from_mujoco_primitive(model, obj_id, geom_type)
        else: 
            # if obj_type=BODY, we assume it has a corresponding mesh defined in the model
            # Construct the trimesh
            mesh_name = obj_name + "_mesh" if mesh_name is None else mesh_name
            mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
            assert mesh_id >= 0, f"Mesh {mesh_name} not found in model."
            obj_mesh = build_trimesh_from_mujoco_mesh(model, mesh_id)
        self.obj_mesh[obj_name] = smooth_mesh(obj_mesh)

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

    def render_taxim_named(self, name, shadow=True, get_depth=True, pcn_add_noise=False, visualize=True):
        '''
        Renders the taxim image for the given object name.
        This function assumes that a contact check has already been made, and thus the object is close enough to the sensor.
        
        :param name: The name of the object to render the taxim image for. Must be a key added using self.add_object_mujoco.
        :param shadow: Whether to render shadows in the image.
        :param get_depth: Whether to return the depth map along with the image.
        :param visualize: Whether to display the image using OpenCV.
        '''
        
        obj_name = name
        wPs, wRs = self.sensor.get_pose()
        wTs = np.eye(4)
        wTs[:3, :3] = wRs
        wTs[:3, 3] = wPs * 1000.0 # change to mm
        wPo, wRo = self.object_links[obj_name].get_pose()
        wTo = np.eye(4)
        wTo[:3, :3] = wRo
        wTo[:3, 3] = wPo * 1000.0 # change to mm

        height_map, gel_map, contact_mask, press_depth, gt_height_map, pcn = self.generateHeightMapWithTransform(wTs, wTo, obj_name, pcn_add_noise=pcn_add_noise)
        heightMap, contact_mask, contact_height = Core.deformApprox(press_depth, height_map, gel_map, contact_mask)
        sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
        sim_img = sim_img if not shadow else shadow_sim_img
        
        # add some gaussian noise to simulate real sensor noise
        noise_sigma = 5
        noise = np.random.normal(0, noise_sigma, sim_img.shape).astype(sim_img.dtype)
        sim_img = cv2.add(sim_img, noise)
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        sim_img = cv2.resize(sim_img, self.resize) if self.resize is not None else sim_img

        hm_return = gt_height_map if get_depth else np.zeros((psp.w, psp.h))
        hm_return = cv2.rotate(hm_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
        hm_return = cv2.resize(hm_return, self.resize) if self.resize is not None else hm_return
        
        if(visualize):
            if not get_depth:
                combined_img = sim_img
            else:
                # repeat height map to 3 channels
                gt_vis = np.repeat(hm_return[:, :, np.newaxis], 3, axis=2)
                div = 1 if np.max(gt_vis) == 0 else np.max(gt_vis)
                gt_vis = (gt_vis / div * 255).astype(np.uint8)
                combined_img = np.concatenate((sim_img, gt_vis), axis=1)
            cv2.imshow("taxim", combined_img)
            cv2.waitKey(1)
        return sim_img, hm_return, pcn
        
    def render_taxim(self, model, data, shadow=True, get_depth=True, pcn_add_noise=False, visualize=True):
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
            hm_return = np.zeros((psp.h, psp.w))
            pcn = np.array([])
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

            # f1: 0.025, deform: 0.025, sim: 0.15
            height_map, gel_map, contact_mask, press_depth, gt_height_map, pcn = self.generateHeightMapWithTransform(wTs, wTo, obj_name, pcn_add_noise=pcn_add_noise)
            heightMap, contact_mask, contact_height = Core.deformApprox(press_depth, height_map, gel_map, contact_mask)
            sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
            sim_img = sim_img if not shadow else shadow_sim_img
        
        # add some gaussian noise to simulate real sensor noise
        noise_sigma = 5
        noise = np.random.normal(0, noise_sigma, sim_img.shape).astype(sim_img.dtype)
        sim_img = cv2.add(sim_img, noise)
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        sim_img = cv2.resize(sim_img, self.resize) if self.resize is not None else sim_img

        hm_return = gt_height_map if get_depth else np.zeros((psp.w, psp.h))
        hm_return = cv2.rotate(hm_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
        hm_return = cv2.resize(hm_return, self.resize) if self.resize is not None else hm_return
        
        if(visualize):
            if not get_depth:
                combined_img = sim_img
            else:
                # repeat height map to 3 channels
                gt_vis = np.repeat(hm_return[:, :, np.newaxis], 3, axis=2)
                div = 1 if np.max(gt_vis) == 0 else np.max(gt_vis)
                gt_vis = (gt_vis / div * 255).astype(np.uint8)
                combined_img = np.concatenate((sim_img, gt_vis), axis=1)
            cv2.imshow("taxim", combined_img)
            cv2.waitKey(1)
        return sim_img, hm_return, pcn
        
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
        grad_mag, grad_dir = Core.generate_normals(heightMap)

        # generate raw simulated image without background
        if not hasattr(self, "_sim_img_r_buf") or self._sim_img_r_buf.shape != (psp.h, psp.w, 3):
            self._sim_img_r_buf = np.zeros((psp.h, psp.w, 3), dtype=np.float32)
        sim_img_r = self._sim_img_r_buf
        sim_img_r.fill(0)
        bins = psp.numBins

        A, xx, yy = self._get_poly_design_cache()
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype(np.int32)
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype(np.int32)

        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]

        A_hw = A.reshape((psp.h,psp.w, -1))
        sim_img_r[:,:,0] = np.einsum('hwk,hwk->hw', A_hw, params_r, optimize=True)
        sim_img_r[:,:,1] = np.einsum('hwk,hwk->hw', A_hw, params_g, optimize=True)
        sim_img_r[:,:,2] = np.einsum('hwk,hwk->hw', A_hw, params_b, optimize=True)

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
        enlarged_mask = dialate_mask.astype(bool)
        contact_mask = enlarged_mask & (~contact_mask.astype(bool))

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = (normMap // pr.discritize_precision).astype(np.int32, copy=False)

        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = ((contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision).astype(np.int32, copy=False)
        if(height_idx.size == 0):
            return sim_img, sim_img
        
        # Shadow calculation
        height_idx_max = int(np.max(height_idx))
        total_height_idx = self.shadowTable.shape[2]

        H, W = psp.h, psp.w
        shadowSim = np.zeros((H, W, 3), dtype=np.float32)

        # Convert to numpy arrays for faster indexing
        x0 = np.asarray(x_coord, dtype=np.int32)
        y0 = np.asarray(y_coord, dtype=np.int32)
        n0 = np.asarray(norm_idx, dtype=np.int32)
        h0 = (np.asarray(height_idx, dtype=np.int32) + 6)

        # Valid indices for shadowTable height dimension
        valid = (h0 >= 0) & (h0 < total_height_idx)
        x0 = x0[valid]; y0 = y0[valid]; n0 = n0[valid]; h0 = h0[valid]

        # Cache trig fan per normal-bin n
        if not hasattr(self, "_fan_trig_cache"):
            self._fan_trig_cache = {}

        # Precompute step indices once per unique profile length if you want, but
        for c in range(3):
            frame = sim_img_r[:, :, c].copy().astype(np.float32)

            # Group pixels by (n, h) to reuse the same v and trig lists
            keys = (n0.astype(np.int64) << 32) | h0.astype(np.int64)
            order = np.argsort(keys)
            keys_sorted = keys[order]

            x_s = x0[order]
            y_s = y0[order]
            n_s = n0[order]
            h_s = h0[order]

            # Iterate groups of same (n,h)
            start_idx = 0
            while start_idx < len(keys_sorted):
                key = keys_sorted[start_idx]
                end_idx = start_idx + 1
                while end_idx < len(keys_sorted) and keys_sorted[end_idx] == key:
                    end_idx += 1

                n = int(n_s[start_idx])
                h = int(h_s[start_idx])

                v = self.shadowTable[c, n, h]
                num_step = len(v)
                if num_step > 1:
                    # cached fan trig for this normal
                    trig = self._fan_trig_cache.get(n, None)
                    if trig is None:
                        d_theta = float(self.direction[n])
                        theta_list = np.arange(d_theta - pr.fan_angle,
                                            d_theta + pr.fan_angle,
                                            pr.fan_precision,
                                            dtype=np.float32)
                        ct_list = np.cos(theta_list).astype(np.float32)
                        st_list = np.sin(theta_list).astype(np.float32)
                        self._fan_trig_cache[n] = (ct_list, st_list)
                    else:
                        ct_list, st_list = trig

                    # Vectorize steps s=1..num_step-1 once for this v
                    s_arr = np.arange(1, num_step, dtype=np.float32)
                    v_arr = np.asarray(v, dtype=np.float32)[1:]  # (S,)

                    # The pixels in this (n,h) group
                    gx = x_s[start_idx:end_idx]
                    gy = y_s[start_idx:end_idx]

                    # Origin heights for occlusion test
                    origin_h = heightMap[gy, gx]  # (P,)

                    # For each fan direction, cast "rays" from all origins at once (vectorized over P and S)
                    for ct, st in zip(ct_list, st_list):
                        xs = (gx[:, None] + pr.shadow_step * s_arr[None, :] * ct).astype(np.int32)
                        ys = (gy[:, None] + pr.shadow_step * s_arr[None, :] * st).astype(np.int32)

                        inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                        if not np.any(inb):
                            continue

                        # Occlusion: origin height > target height
                        # Only evaluate where in-bounds to avoid invalid indexing
                        xs_in = xs[inb]
                        ys_in = ys[inb]

                        # Map inb mask to a flat list of (origin_index, step_index)
                        # We need origin heights aligned with those flat coords:
                        # origin index is row index of xs/ys in the (P,S) grid.
                        # Compute it from inb positions:
                        #   flat indices correspond to row-major flattening, so:
                        #     origin_idx = flat_idx // S
                        S = s_arr.shape[0]
                        flat_inb = np.flatnonzero(inb.ravel())
                        origin_idx = (flat_inb // S).astype(np.int32)

                        occ = origin_h[origin_idx] > heightMap[ys_in, xs_in]
                        if not np.any(occ):
                            continue

                        # Apply minimum update with corresponding v[s]
                        # step index: flat_idx % S gives s-1 index into v_arr
                        step_idx = (flat_inb % S).astype(np.int32)

                        xs_hit = xs_in[occ]
                        ys_hit = ys_in[occ]
                        vv_hit = v_arr[step_idx[occ]]

                        np.minimum.at(frame, (ys_hit, xs_hit), vv_hit)

                start_idx = end_idx

            shadowSim[:, :, c] = ndimage.gaussian_filter(frame, sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim + self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32), (pr.kernel_size, pr.kernel_size), 0)
        return sim_img, shadow_sim_img

    def generateHeightMapWithTransform(self, wTs, wTo, obj_name, pressing_mm_max = 3.0, pcn_add_noise=False):
        """
        Generate the height map by interacting the object with the gelpad model.

        wTs: world to sensor transformation matrix
        wTo: world to object transformation matrix
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # load dome-shape gelpad model
        gel_map = self.gel_map.copy()
        heightMap = np.zeros((psp.h,psp.w))

        # calculate sTo: object-in-sensor-frame transform
        sTw = invert_homogeneous_matrix(wTs)
        sTo = sTw @ wTo

        # Rasterization method takes ~3x more time than the original pointcloud method
        # With bbox culling, 2x speedup
        # with njit, 10x faster than original pc approach
        
        # Rasterize the depth of the object in sensor frame
        zbuf = Core.rasterize_depth_from_trimesh(
            self.obj_mesh[obj_name],
            sTo,
            psp.h,
            psp.w,
            psp.pixmm,
        )
        heightMap = Core.heightmap_from_zbuf(zbuf, psp.pixmm)
        n_points = 5000
        pcn = Core.pointcloud_from_zbuf_with_normals(zbuf, psp.pixmm, n_points=n_points, roughness_enable=pcn_add_noise)
        # assert pcn.shape[0] == n_points, "Pointcloud does not have the expected number of points."

        # pressing depth in pixel
        valid = np.isfinite(zbuf)          # pixels where mesh projects
        if np.any(valid):
            min_z = np.min(zbuf[valid])    # most negative z (deepest), or could be >0 if mesh is above gel
            pressing_height_mm = min(pressing_mm_max, max(0.0, -min_z))
        else:
            pressing_height_mm = 0.0
            
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
        return zq, gel_map, contact_mask, pressing_height_mm, heightMapBlur, pcn
    
    def _get_poly_design_cache(self):
        cache_key = (psp.w, psp.h)
        if getattr(self, "_poly_design_cache_key", None) != cache_key:
            grid_x, grid_y = np.meshgrid(range(psp.w), range(psp.h))
            flat_x = grid_x.flatten()
            flat_y = grid_y.flatten()
            # match original dtype behavior: np.array([...]).T -> float64
            self._poly_design_cache = np.array(
                [flat_x * flat_x,
                flat_y * flat_y,
                flat_x * flat_y,
                flat_x,
                flat_y,
                np.ones(psp.w * psp.h)]
            ).T
            self._poly_design_cache_key = cache_key
            self._xy_grid_cache = (grid_x, grid_y)  # also reuse xx,yy later
        return self._poly_design_cache, self._xy_grid_cache[0], self._xy_grid_cache[1]
            