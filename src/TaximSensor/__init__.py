from dataclasses import dataclass
from pathlib import Path
import warnings
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
from norm2tex.normals import (
    BUMP_DIRECTION,
    approximate_height_map_from_normal_map,
    pseudo_height_to_uint8_image,
    rasterize_and_apply_uv_normals,
)
from TaximSensor.helpers import (
    _penetration_stats_between_body_and_geom, 
    invert_homogeneous_matrix, 
    build_trimesh_from_mujoco_mesh, 
    build_trimesh_from_mujoco_primitive, 
    bgr_to_rgb, 
    rgb_to_bgr, 
    build_trimesh_with_uvs_from_mujoco_mesh,
    )
from norm2tex.timing import (
    timed,
    print_timings,
    reset_timings
)
__version__ = "0.1"  # Source of truth for mujoco-taxim's version

_exported_dunders = {
    "__version__",
}


def _depth_map_path_from_normal_map(normal_map_path: str | Path) -> Path:
    normal_path = Path(normal_map_path)
    depth_name = normal_path.name.replace("normal", "depth")
    if depth_name == normal_path.name:
        depth_name = f"{normal_path.stem}_depth{normal_path.suffix}"
    return normal_path.with_name(depth_name)


def _pseudo_height_from_depth_image(
    depth_image: np.ndarray,
    bump_direction: BUMP_DIRECTION,
) -> np.ndarray:
    if depth_image.ndim == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    normalized = depth_image.astype(np.float32)
    if np.issubdtype(depth_image.dtype, np.integer):
        normalized /= np.iinfo(depth_image.dtype).max
    else:
        max_v = float(np.nanmax(normalized)) if normalized.size else 0.0
        if max_v > 1.0:
            normalized /= 255.0
    if bump_direction == BUMP_DIRECTION.ZERO_CENTERED:
        return (normalized * 2.0 - 1.0).astype(np.float32)
    return normalized.astype(np.float32)

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
    def __init__(
        self,
        sensor_type="digit",
        bg_file=None,
        gelmap_file=None,
        bg_index=0,
        resize=None,
        preprocess_bg=True,
        texture_bump_scale_mm=0.05,
        raster_backend="cuda",
        cuda_device=0,
    ):
        '''
        Initialize the simulator.
        1) load the calibration files,
        2) generate shadow table from shadow masks
        3) load the gelpad model

        :param self: Description
        :param data_folder: root path to calibration data
        :param gelpad_model_path: path to the gelpad model numpy file
        ''' 
        if sensor_type != "digit" and sensor_type != "gelsight_r1.5":
            raise NotImplementedError("Currently only digit and gelsight_r1.5 sensors are supported.")

        self.sensor_type = sensor_type
        self.obj_mesh = {}
        self.obj_raster_vertices_h = {}
        self.obj_raster_faces = {}
        self.obj_raster_stats = {}
        self.object_links = {}
        self.object_body_ids = {}
        self.saved=False 
        # polytable
        calib_data = f"{sensor_type}/polycalib.npz"
        self.calib_data = CalibData(calib_data)
        self.resize=resize
        self.texture_bump_scale_mm = texture_bump_scale_mm
        if raster_backend not in {"cpu", "cuda"}:
            raise ValueError("raster_backend must be either 'cpu' or 'cuda'")
        self.requested_raster_backend = raster_backend
        self.raster_backend = raster_backend
        self._cuda_raster = None
        if raster_backend == "cuda":
            from TaximSensor.cuda import CudaRasterBackend, cuda_device_available

            cuda_available, unavailable_reason = cuda_device_available(cuda_device)
            if cuda_available:
                self._cuda_raster = CudaRasterBackend(
                    height=psp.h,
                    width=psp.w,
                    pixmm=psp.pixmm,
                    device=cuda_device,
                )
            else:
                self.raster_backend = "cpu"
                warnings.warn(
                    f"CUDA backend requested but unavailable: {unavailable_reason}. "
                    "Falling back to the CPU backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # raw calibration data, here only used for background
        if bg_file is None:
            data_file = read_calib_np(f"{sensor_type}/bg_set.npz")
        else:
            data_file = np.load(bg_file, allow_pickle=True)
        self.data_file = data_file['f0']
        self.bgs = []
        self.bgs_rot = []
        for i, df in enumerate(self.data_file):
            self.data_file[i] = rgb_to_bgr(df).copy() # Do a channel flip at init to avoid confusion down the line
        for i in range(self.data_file.shape[0]):
            self.f0 = self.data_file[i]
            if preprocess_bg:
                self.bgs.append(self.processInitialFrame())
            else:
                self.bgs.append(self.f0.copy()) # So this keeps the bg looking exactly the same... procInitFrame just applies some gaussian filtering to make the bg smoother
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

        if gelmap_file is None:
            self.gel_map = read_calib_np("gelmap5.npy")
        else:
            self.gel_map = read_calib_np(gelmap_file)
            assert self.gel_map.shape == (480, 640), "Gelmap shape should be (480, 640) to stay consistent with original gelmap."
            assert self.gel_map.max() <= 169.5, "Gelmap max should not exceed 169.5 to stay consistent with original gelmap."
            assert self.gel_map.min() >= 122.0, "Gelmap min should not be less than 122.0 to stay consistent with original gelmap."
        self.gel_map = cv2.GaussianBlur(self.gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        if self._cuda_raster is not None:
            self._cuda_raster.configure_frame_pipeline(
                gel_map=self.gel_map,
                background=self.bg_proc,
                grad_r=self.calib_data.grad_r,
                grad_g=self.calib_data.grad_g,
                grad_b=self.calib_data.grad_b,
                shadow_directions=self.direction,
                shadow_table=self.shadowTable,
                fan_angle=pr.fan_angle,
                fan_precision=pr.fan_precision,
            )

    def get_current_bg(self):
        '''
        Returns the current background, rotated to Portrait, in RGB.
        '''
        return bgr_to_rgb(self.bg_proc_rot)
    
    def change_bg(self, bg_index):
        if bg_index > len(self.bgs)-1:
            print("Warning: bg_index exceeds the number of available backgrounds. No change made.")
            return
        self.f0 = self.data_file[bg_index]
        self.bg_proc = self.bgs[bg_index]
        self.bg_proc_rot = self.bgs_rot[bg_index]
        self.bg_index = bg_index
        if self._cuda_raster is not None:
            self._cuda_raster.update_background(self.bg_proc)

    def add_geom_mujoco(
        self,
        geom_name: str,
        model,
        data,
        mesh_name: str,
        normal_map_path: str = None,
        texture_map_direction=BUMP_DIRECTION.ZERO_CENTERED,
    ):
        """
        Add a mjGEOM to the list of objects to be tracked by the sensor.
        Other object types are not supported, as a 3D object in mujoco necessarily requires a geom tag, and we are
            only interested in the pose of that geom for rendering, not the body.
        Primitive geometries are handled accordingly.

        Since it requires the corresponding geom's pose in the simulation, at least one mj_step should be called
        before this function.

        :param geom_name: str
            Name of the geom to be added. This is defined as a mujoco <geom> tag.
        :param model: mjModel
        :param data: mjData
        :param mesh_name: str
            Name of the ground truth mesh to be used for the object, i.e. the one to be simulated.
        """
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        obj_type = mj.mjtObj.mjOBJ_GEOM
        uvs = None
        raster_data = None

        assert geom_id >= 0, f"Geometry {geom_name} not found in model."
        # Keep track of body id for contact checking
        self.object_body_ids[geom_name] = model.geom_bodyid[geom_id]
        self.object_links[geom_name] = Link(
            geom_id, obj_type, data, model, geom_name
        )

        # if obj_type=GEOM, we need to check if it is a mesh or a primitive
        geom_type = model.geom_type[geom_id]

        if(geom_type == mj.mjtGeom.mjGEOM_MESH):
            # if mesh, use the mesh name for creating the trimesh
            # Construct the trimesh
            mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
            assert mesh_id >= 0, f"Mesh {mesh_name} not found in model."
            try:
                obj_mesh, uvs, raster_data = build_trimesh_with_uvs_from_mujoco_mesh(
                    model,
                    mesh_id,
                    return_raster_data=True,
                )
            except ValueError:
                print(f"WARNING: Mesh {mesh_name} does not contain UVs. Falling back to non-UV mesh.")
                obj_mesh = build_trimesh_from_mujoco_mesh(model, mesh_id)
        else:
            obj_mesh = build_trimesh_from_mujoco_primitive(model, geom_id, geom_type)

        if raster_data is None:
            vertices_mm = np.asarray(obj_mesh.vertices, dtype=np.float32) * np.float32(1000.0)
            self.obj_raster_vertices_h[geom_name] = np.concatenate(
                (vertices_mm, np.ones((len(vertices_mm), 1), dtype=np.float32)),
                axis=1,
            )
            self.obj_raster_faces[geom_name] = obj_mesh.faces.astype(
                np.int32,
                copy=False,
            )
        else:
            self.obj_raster_vertices_h[geom_name] = raster_data.vertices_h
            self.obj_raster_faces[geom_name] = raster_data.faces
        self.obj_raster_stats[geom_name] = {
            "vertices": len(self.obj_raster_vertices_h[geom_name]),
            "faces": len(self.obj_raster_faces[geom_name]),
            "face_corners": len(uvs) if uvs is not None else len(obj_mesh.vertices),
        }
        if normal_map_path is not None and uvs is not None:
            # Load all the normal map related data at add
            if not hasattr(self, "obj_uvs"):
                self.obj_uvs = {}
            if not hasattr(self, "obj_pseudo_height"):
                self.obj_pseudo_height = {}
            if not hasattr(self, "obj_trimesh"):
                self.obj_trimesh = {}
            if not hasattr(self, "obj_uv_tris"):
                self.obj_uv_tris = {}
            if not hasattr(self, "obj_normal_tris"):
                self.obj_normal_tris = {}
            self.obj_trimesh[geom_name] = obj_mesh 
            self.obj_uvs[geom_name] = uvs
            if raster_data is None:
                faces_i = self.obj_raster_faces[geom_name]
                self.obj_uv_tris[geom_name] = np.asarray(uvs, dtype=np.float32)[faces_i]
                vertex_normals = np.asarray(obj_mesh.vertex_normals, dtype=np.float32)
                self.obj_normal_tris[geom_name] = vertex_normals[faces_i]
            else:
                self.obj_uv_tris[geom_name] = raster_data.uv_tris
                self.obj_normal_tris[geom_name] = raster_data.normal_tris
            depth_map_path = _depth_map_path_from_normal_map(normal_map_path)
            if depth_map_path.exists():
                cached_depth = cv2.imread(str(depth_map_path), cv2.IMREAD_UNCHANGED)
                if cached_depth is None:
                    raise ValueError(f"Could not read cached depth map '{depth_map_path}'.")
                self.obj_pseudo_height[geom_name] = _pseudo_height_from_depth_image(
                    cached_depth,
                    texture_map_direction,
                )
                print(f"Loaded cached pseudo-height depth map from {depth_map_path}")
            else:
                self.obj_pseudo_height[geom_name] = approximate_height_map_from_normal_map(
                    normal_map_path,
                    # Blender bakes tangent-space normals in the OpenGL convention.
                    # Flipping the green channel here avoids reconstructing each bump
                    # as a split peak/valley response.
                    invert_y=True,
                    bump_direction=texture_map_direction,
                )
                depth_vis_gray = pseudo_height_to_uint8_image(
                    self.obj_pseudo_height[geom_name]
                )
                cv2.imwrite(str(depth_map_path), depth_vis_gray)
                print(f"Saved pseudo-height depth map to {depth_map_path}")

        # self.obj_mesh[geom_name] = self.obj_trimesh[geom_name] if normal_map_path is not None else obj_mesh
        self.obj_mesh[geom_name] = obj_mesh
        if self._cuda_raster is not None:
            textured = hasattr(self, "obj_pseudo_height") and geom_name in self.obj_pseudo_height
            self._cuda_raster.register_object(
                geom_name,
                self.obj_raster_vertices_h[geom_name],
                self.obj_raster_faces[geom_name],
                uv_tris=self.obj_uv_tris[geom_name] if textured else None,
                normal_tris=self.obj_normal_tris[geom_name] if textured else None,
                pseudo_height=self.obj_pseudo_height[geom_name] if textured else None,
            )

    def get_raster_stats(self, obj_name: str) -> dict[str, int]:
        """Return geometry counts used by the CPU or CUDA raster backend."""
        try:
            return dict(self.obj_raster_stats[obj_name])
        except KeyError as exc:
            raise KeyError(f"Raster object {obj_name!r} has not been registered") from exc

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
    
    def set_sensor_pad_geom(self, geom_name):
        self.sensor_pad_geom_name = geom_name
    
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

    def render_taxim_named(self, name, shadow=True, get_depth=True, img_noise_sigma=5, pcn_add_noise=False, visualize=True, cycle_bg=True):
        '''
        Renders the taxim image for the given object name, and returns the simulated image, ground truth height map, and point cloud.
        This function assumes that a contact check has already been made, and thus the object is close enough to the sensor.

        Returns the rendered taxim image in RGB format.
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
        noise_sigma = img_noise_sigma
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
            cv2.imshow("taxim_" + self.sensor_pad_geom_name, combined_img)
            cv2.waitKey(1)
        if cycle_bg:
            self.change_bg((self.bg_index + 1) % self.bg_len)
        # for gelsight OFR, bgr_to_rgb(sim_img); for Digit, not needed for some reason 
        return bgr_to_rgb(sim_img), hm_return, pcn
    
    def render_blank_taxim(self, shadow=True):
        gel_map = self.gel_map
        height_map = np.zeros((psp.h, psp.w))
        press_depth = 0.0
        contact_mask = height_map > gel_map
        # heightMap, contact_mask, contact_height = Core.deformApprox(press_depth, height_map, gel_map, contact_mask)
        heightMap = np.zeros((psp.h, psp.w))
        contact_height = np.zeros((psp.h, psp.w))
        sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
        sim_img = sim_img if not shadow else shadow_sim_img
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        sim_img = cv2.resize(sim_img, self.resize) if self.resize is not None else sim_img

        return bgr_to_rgb(sim_img)

    def render_taxim(
        self,
        model,
        data,
        shadow=True,
        get_depth=True,
        img_noise_sigma=5,
        pcn_add_noise=False,
        visualize=True,
        cycle_bg=False,
    ):
        with timed("taxim_total"):
            result = self._render_taxim_impl(
                model=model,
                data=data,
                shadow=shadow,
                get_depth=get_depth,
                img_noise_sigma=img_noise_sigma,
                pcn_add_noise=pcn_add_noise,
                visualize=visualize,
                cycle_bg=cycle_bg,
            )
        print_timings()
        return result

    def _render_taxim_impl(
        self,
        model,
        data,
        shadow=True,
        get_depth=True,
        img_noise_sigma=5,
        pcn_add_noise=False,
        visualize=True,
        cycle_bg=False,
    ):
        '''
        Renders the taxim image based on the current mujoco state, and returns the simulated image, ground truth height map, and point cloud.

        Returns the rendered taxim image in RGB format.
        '''

        '''
        Check if there are bodies in contact with the sensor
        '''
        bodies_in_contact = []
        for geom_name, body_id in self.object_body_ids.items():
            _, max_pen, _ = _penetration_stats_between_body_and_geom(model, data, body_id, self.sensor_pad_geom_name)
            if max_pen > 0:
                bodies_in_contact.append(geom_name)
        
        debug_base_height = np.zeros((psp.h, psp.w), dtype=np.float32)
        debug_bumpy_height = np.zeros((psp.h, psp.w), dtype=np.float32)
        debug_contact_mask = np.zeros((psp.h, psp.w), dtype=bool)

        if len(bodies_in_contact) == 0:
            sim_img = self.bg_proc.astype(np.float32)
            hm_return = np.zeros((psp.h, psp.w))
            pcn = np.array([])
            gt_height_map = np.zeros((psp.h, psp.w))
            overlay = np.zeros((psp.h, psp.w))
        else:
            # We assume that only 1 object is in contact at any given moment
            obj_name = bodies_in_contact[0]
            wPs, wRs = self.sensor.get_pose()
            wTs = np.eye(4)
            wTs[:3, :3] = wRs
            wTs[:3, 3] = wPs * 1000.0 # change to mm
            wPo, wRo = self.object_links[obj_name].get_pose()
            wTo = np.eye(4)
            wTo[:3, :3] = wRo
            wTo[:3, 3] = wPo * 1000.0 # change to mm

            if self._cuda_raster is not None:
                sTo = invert_homogeneous_matrix(wTs) @ wTo
                with timed("cuda_pipeline_wall"):
                    sim_img, gt_height_map, overlay = self._cuda_raster.render_frame(
                        obj_name,
                        sTo,
                        bump_scale_mm=self.texture_bump_scale_mm,
                        pressing_mm_max=3.0,
                        contact_scale=pr.contact_scale,
                        pyramid_kernel_sizes=tuple(pr.pyramid_kernel_size),
                        final_kernel_size=pr.kernel_size,
                        shadow=shadow,
                        shadow_depth_min=self.shadow_depth[0],
                        height_precision=pr.height_precision,
                        direction_precision=pr.discritize_precision,
                        shadow_step=pr.shadow_step,
                        shadow_sigma=pr.sigma,
                    )
                pcn = None
            else:
                with timed("hm_total"):
                    height_map, gel_map, contact_mask, press_depth, gt_height_map, pcn, overlay = self.generateHeightMapWithTransform(wTs, wTo, obj_name, pcn_add_noise=pcn_add_noise)
                    debug_bumpy_height = np.asarray(height_map, dtype=np.float32)
                    debug_contact_mask = np.asarray(contact_mask, dtype=bool)
                    debug_base_height = np.clip(debug_bumpy_height - np.asarray(overlay, dtype=np.float32), 0.0, None)
                with timed("deform_total"):
                    heightMap, contact_mask, contact_height = Core.deformApprox(press_depth, height_map, gel_map, contact_mask)
                with timed("sim_total"):
                    sim_img, shadow_sim_img = self.simulating(heightMap, contact_mask, contact_height, shadow=shadow)
                    sim_img = sim_img if not shadow else shadow_sim_img

        # add some gaussian noise to simulate real sensor noise
        # noise_sigma = img_noise_sigma
        # noise = np.random.normal(0, noise_sigma, sim_img.shape).astype(sim_img.dtype)
        noise_sigma = img_noise_sigma
        grain_size = 4  # bigger = chunkier noise

        h, w = sim_img.shape[:2]
        c = 1 if sim_img.ndim == 2 else sim_img.shape[2]

        small_h = max(1, h // grain_size)
        small_w = max(1, w // grain_size)

        # one noise value per coarse block
        coarse_noise = np.random.normal(0, noise_sigma, (small_h, small_w, c)).astype(np.float32)

        # upscale with nearest-neighbor so blocks stay visible
        noise = cv2.resize(coarse_noise, (w, h), interpolation=cv2.INTER_NEAREST)

        sim_img = cv2.add(sim_img, noise)
        sim_img  = cv2.rotate(np.clip(np.rint(sim_img), 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        sim_img = cv2.resize(sim_img, self.resize) if self.resize is not None else sim_img

        hm_return = gt_height_map if get_depth else np.zeros((psp.w, psp.h))
        hm_return = cv2.rotate(hm_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
        hm_return = cv2.resize(hm_return, self.resize) if self.resize is not None else hm_return

        ol_return = overlay if get_depth else np.zeros((psp.w, psp.h))
        ol_return = cv2.rotate(ol_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ol_return = cv2.resize(ol_return, self.resize) if self.resize is not None else ol_return

        if(visualize):
            if not get_depth:
                combined_img = sim_img
            else:
                # repeat height map to 3 channels
                gt_vis = np.repeat(hm_return[:, :, np.newaxis], 3, axis=2)
                div = 1 if np.max(gt_vis) == 0 else np.max(gt_vis)
                gt_vis = (gt_vis / div * 255).astype(np.uint8)

                overlay_vis = np.repeat(ol_return[:, :, np.newaxis], 3, axis=2)
                div_ov = 1 if np.max(overlay_vis) == 0 else np.max(overlay_vis)
                overlay_vis = (overlay_vis / div_ov * 255).astype(np.uint8)

                combined_img = np.concatenate((sim_img, gt_vis, overlay_vis), axis=1)
            cv2.imshow("taxim_"+self.sensor_pad_geom_name, combined_img)
            cv2.waitKey(1)

        # if get_depth:
        #     debug_base = cv2.rotate(debug_base_height, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     debug_bumpy = cv2.rotate(debug_bumpy_height, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     debug_mask = cv2.rotate(debug_contact_mask.astype(np.uint8) * 255, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     debug_overlay = cv2.rotate(np.asarray(overlay, dtype=np.float32), cv2.ROTATE_90_COUNTERCLOCKWISE)

        #     if self.resize is not None:
        #         debug_base = cv2.resize(debug_base, self.resize)
        #         debug_bumpy = cv2.resize(debug_bumpy, self.resize)
        #         debug_mask = cv2.resize(debug_mask, self.resize, interpolation=cv2.INTER_NEAREST)
        #         debug_overlay = cv2.resize(debug_overlay, self.resize)

        #     base_max = float(np.max(debug_bumpy)) if np.max(debug_bumpy) > 0 else 1.0
        #     overlay_abs_max = float(np.max(np.abs(debug_overlay))) if np.max(np.abs(debug_overlay)) > 0 else 1.0

        #     debug_base_vis = (np.clip(debug_base / base_max, 0.0, 1.0) * 255).astype(np.uint8)
        #     debug_bumpy_vis = (np.clip(debug_bumpy / base_max, 0.0, 1.0) * 255).astype(np.uint8)
        #     debug_overlay_vis = (
        #         np.clip((debug_overlay / (2.0 * overlay_abs_max)) + 0.5, 0.0, 1.0) * 255
        #     ).astype(np.uint8)
            # import time
            # if time.time() % 2 < 0.1:
            #     uuid = int(time.time())
            #     cv2.imwrite(f"{uuid}_debug_base_raw.png", debug_base_vis)
            #     cv2.imwrite(f"{uuid}_debug_bumpy_raw.png", debug_bumpy_vis)
            #     cv2.imwrite(f"{uuid}_debug_overlay_signed.png", debug_overlay_vis)
            #     cv2.imwrite(f"{uuid}_debug_contact_mask.png", debug_mask)

        if cycle_bg:
            self.change_bg((self.bg_index + 1) % self.bg_len)
        return bgr_to_rgb(sim_img), hm_return, pcn
        
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

    def generateHeightMapWithTransform(self, wTs, wTo, obj_name, pressing_mm_max = 3.0, return_pcn=False, pcn_add_noise=False):
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
        gel_map = self.gel_map

        # calculate sTo: object-in-sensor-frame transform
        sTw = invert_homogeneous_matrix(wTs)
        sTo = sTw @ wTo

        has_uv_texture = hasattr(self, "obj_uvs") and obj_name in self.obj_uvs
        vertices_h = self.obj_raster_vertices_h[obj_name]
        faces = self.obj_raster_faces[obj_name]

        if self._cuda_raster is not None:
            timing_label = "hm_texture_total" if has_uv_texture else "hm_raster_total"
            with timed(timing_label):
                heightMap, overlay, zbuf = self._cuda_raster.rasterize(
                    obj_name,
                    sTo,
                    bump_scale_mm=self.texture_bump_scale_mm,
                )
        elif has_uv_texture:
            # The UV pass already produces the undisplaced z-buffer. Reuse it
            # instead of rasterizing the same mesh once in Core and once in
            # norm2tex.
            with timed("hm_texture_total"):
                heightMap, overlay, zbuf = rasterize_and_apply_uv_normals(
                    self.obj_trimesh[obj_name],
                    self.obj_uvs[obj_name],
                    self.obj_pseudo_height[obj_name],
                    sTo,
                    (psp.h, psp.w),
                    psp.pixmm,
                    bump_scale_mm=self.texture_bump_scale_mm,
                    uv_tris_cache=self.obj_uv_tris[obj_name],
                    normal_tris_cache=self.obj_normal_tris[obj_name],
                    vertices_h_cache=vertices_h,
                    faces_cache=faces,
                )
        else:
            zbuf = Core.rasterize_depth_from_trimesh(
                self.obj_mesh[obj_name],
                sTo,
                psp.h,
                psp.w,
                psp.pixmm,
                vertices_h_cache=vertices_h,
                faces_cache=faces,
            )
            heightMap = Core.heightmap_from_zbuf(zbuf, psp.pixmm)
            overlay = np.zeros_like(heightMap)
        
        n_points = 5000
        pcn = None
        if return_pcn:
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
        max_g = float(gel_map.max())
        max_o = float(heightMap.max())
        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)

        # get the contact area 
        contact_mask = heightMap > gel_map # heightMap > gel_map
        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.where(contact_mask, heightMap, gel_map)
        heightMapBlur = cv2.GaussianBlur(
            heightMap.astype(np.float32) / max(max_o, 1e-8),
            (5, 5),
            0,
        )

        return zq, gel_map, contact_mask, pressing_height_mm, heightMapBlur, pcn, overlay
    
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
            
