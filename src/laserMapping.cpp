// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

using namespace std;

/** fisheye image **/
cv::Mat img;
int img_idx = 0;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true, lins_save_en = false;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, cloud_deskewed_topic, odometry_topic, body_frame, odom_frame;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, filter_size_submap_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited, deskew_enabled = true;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
double keyframe_dist = 1.0, keyframe_rot = 0.0, keyframe_time = 0.0;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> lidar_to_body_T(3, 0.0);
vector<double> lidar_to_body_R(9, 0.0);

deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<PointVector> feats_window_buffer;
deque<PointVector> feats_window_buffer2;
deque<std::vector<BoxPointType>> box_needrm_window_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr featsSubMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_lidarbody(new PointCloudXYZI());   // feats_undistort --> feats_down_lidarbody --> feats_down_world
PointCloudXYZI::Ptr feats_down_submap_body(new PointCloudXYZI()); // |--> featsSubMap
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
pcl::VoxelGrid<PointType> downSizeFilterSubMap; // submap使用的降采样

KD_TREE<PointType> ikdtree;
KD_TREE<PointType> ikdtree_submap;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);  // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);   // R lidar to imu (imu = r * lidar + t)
V3D Lidar_T_wrt_BODY(Zero3d); // T lidar to body (body = r * lidar + t)
M3D Lidar_R_wrt_BODY(Eye3d);  // R lidar to body (body = r * lidar + t)
V3D IMU_T_wrt_BODY(Zero3d);   // T imu to body (body = r * imu + t)
M3D IMU_R_wrt_BODY(Eye3d);    // R imu to body (body = r * imu + t)
V3D BODY_T_wrt_IMU(Zero3d);   // T body to imu (imu = r * body + t)
M3D BODY_R_wrt_IMU(Eye3d);    // R body to imu (imu = r * body + t)

/*submap*/
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZRGB::Ptr pcl_wait_save_rgb(new PointCloudXYZRGB());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save_infov(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_submap(new PointCloudXYZI());

// pcl::ExtractIndices<PointType> extract;
// std::deque<int> inliers_end_vec;
std::deque<PointCloudXYZI::Ptr> pcl_submap_vec;
// unsigned int window_start_idx, window_end_idx, total_size;
int FEATS_WINDOW_SIZE = 10;
ofstream foutLINS;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
V3D state_pos_world_body;
M3D state_rot_world_body;
vect3 pos_lid;
std::string LINS_RESULT_PATH;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped, LastKeyFrameOdom, keyFrameOdom;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// 将imu下的位姿转到body系下
void getCurPose(state_ikfom cur_state)
{
    V3D state_pos(state_point.pos(0), state_point.pos(1), state_point.pos(2));
    M3D state_rot = state_point.rot.toRotationMatrix();
    V3D b0_T_ij = IMU_R_wrt_BODY * state_pos + IMU_T_wrt_BODY;
    M3D b0_R_ij = IMU_R_wrt_BODY * state_rot;
    state_pos_world_body = b0_R_ij * BODY_T_wrt_IMU + b0_T_ij;
    state_rot_world_body = b0_R_ij * BODY_R_wrt_IMU;
    // Eigen::Quaterniond q(state_rot_world_body);

    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    // Eigen::Vector3d eulerAngle = state_point.rot.toRotationMatrix().eulerAngles(2, 1, 0); //  yaw pitch roll  单位：弧度
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw

    // transformTobeMapped[0] = eulerAngle(2); //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    // transformTobeMapped[1] = eulerAngle(1); //  pitch
    // transformTobeMapped[2] = eulerAngle(0); //  yaw
    // transformTobeMapped[3] = state_pos(0);  //  x
    // transformTobeMapped[4] = state_pos(1);  //   y
    // transformTobeMapped[5] = state_pos(2);  // z
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void IntensitypointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D pt_imu(pi->x, pi->y, pi->z);
    // V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
    V3D p_global(state_rot_world_body * (IMU_R_wrt_BODY * pt_imu + IMU_T_wrt_BODY) + state_pos_world_body);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

Params params;
/** use camera to colorize the published point cloud **/
void RGBpointBodyToWorld(PointType const *const pi, PointRgbType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);

    /** project to the fisheye camera and get the rgb value **/
    /** extrinsic transformation **/
    Eigen::Vector3d point_vec;
    Eigen::Vector3d point_trans_vec;
    point_vec << pi->x, pi->y, pi->z;
    point_trans_vec = params.R * point_vec + params.translation;

    /** intrinsic transformation **/
    Eigen::Vector2d uv_vec;
    Eigen::Matrix2d affine_inv;
    double theta, uv_radius, xy_radius;
    theta = acos(point_trans_vec(2) / sqrt((point_trans_vec(0) * point_trans_vec(0)) + (point_trans_vec(1) * point_trans_vec(1)) + (point_trans_vec(2) * point_trans_vec(2))));
    uv_radius = params.a_(0) + params.a_(1) * theta + params.a_(2) * pow(theta, 2) + params.a_(3) * pow(theta, 3) + params.a_(4) * pow(theta, 4);
    xy_radius = sqrt(point_trans_vec(1) * point_trans_vec(1) + point_trans_vec(0) * point_trans_vec(0));
    uv_vec = {uv_radius / xy_radius * point_trans_vec(0) + params.uv_0(0), uv_radius / xy_radius * point_trans_vec(1) + params.uv_0(1)};
    affine_inv.row(0) << params.affine(1, 1) / (params.affine(0, 0) * params.affine(1, 1) - params.affine(1, 0) * params.affine(0, 1)),
        -params.affine(0, 1) / (params.affine(0, 0) * params.affine(1, 1) - params.affine(1, 0) * params.affine(0, 1));
    affine_inv.row(1) << -params.affine(1, 0) / (params.affine(0, 0) * params.affine(1, 1) - params.affine(1, 0) * params.affine(0, 1)),
        params.affine(0, 0) / (params.affine(0, 0) * params.affine(1, 1) - params.affine(1, 0) * params.affine(0, 1));
    uv_vec = affine_inv * uv_vec;

    //    std::cout << point_vec << "\n" << point_trans_vec << std::endl;
    //    std::cout << "theta: " << theta << " xy_radius: " << xy_radius << std::endl;
    //    std::cout << "u: " << uv_vec[0] << " v: " << uv_vec[1] << " radius: " << uv_radius << std::endl;

    // if image_point is within the frame
    if (0 <= uv_vec[0] && uv_vec[0] < img.rows && 0 <= uv_vec[1] && uv_vec[1] < img.cols)
    {
        if (uv_radius > 400 & uv_radius < 1000)
        {
            po->b = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[0];
            po->g = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[1];
            po->r = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[2];
            //            std::cout << "Use img: " << img_idx << std::endl;
        }
    }
    else
    {
        po->b = 255;
        po->g = 255;
        po->r = 255;
    }

    po->b = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

// 在拿到eskf前馈结果后，动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
BoxPointType LocalMap_Points;      // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false; // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    // X轴分界点转换到w系下，好像没有用到
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    // global系下lidar位置
    V3D pos_LiD = pos_lid;
    // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
    if (!Localmap_Initialized)
    { // 系统起始需要初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（例如1.5*300m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    // 不需要挪动就直接退回了
    if (!need_move)
        return;
    // 否则需要计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    // 新的局部地图盒子边界点
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        // 与包围盒最小值边界点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    // 使用Boxs删除指定盒内的点
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void img_cbk(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv_ptr->image.copyTo(img);
        img_idx++;
        std::cout << "Subcribe img: " << img_idx << std::endl;
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;            // 需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample; // 加入ikd-tree时，不需要降采样的点云
    PointVector PointForLocalmap;
    std::vector<BoxPointType> box_needrm;
    PointToAdd.reserve(feats_down_size);            // 构建的地图点
    PointNoNeedDownsample.reserve(feats_down_size); // 构建的地图点，不需要降采样的点云
    PointForLocalmap.reserve(feats_down_size);
    // 根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        // 转换到世界坐标系下
        pointBodyToWorld(&(feats_down_lidarbody->points[i]), &(feats_down_world->points[i]));
        // 判断是否有关键点需要加到地图中
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i]; // 获取附近的点云
            bool need_add = true;                               // 是否需要加入到地图中
            BoxPointType Box_of_Point;                          // 点云所在的包围盒
            PointType downsample_result, mid_point;             // 降采样结果，中点
            // 计算该点所属的体素
            Box_of_Point.vertex_min[0] = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min;
            Box_of_Point.vertex_max[0] = Box_of_Point.vertex_min[0] + filter_size_map_min;
            Box_of_Point.vertex_min[1] = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min;
            Box_of_Point.vertex_max[1] = Box_of_Point.vertex_min[1] + filter_size_map_min;
            Box_of_Point.vertex_min[2] = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min;
            Box_of_Point.vertex_max[2] = Box_of_Point.vertex_min[2] + filter_size_map_min;
            box_needrm.push_back(Box_of_Point);
            // filter_size_map_min是地图体素降采样的栅格边长，设为0.1m
            // mid_point即为该特征点所属的栅格的中心点坐标
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            // 当前点与box中心的距离
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            // 判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                // 若三个方向距离都大于地图栅格半轴长，无需降采样
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            // 判断当前点的 NUM_MATCH_POINTS 个邻近点 与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) // 若邻近点数小于NUM_MATCH_POINTS，则直接跳出，添加到PointToAdd中
                    break;
                // 如果存在邻近点到中心的距离小于当前点到中心的距离，则不需要添加当前点
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]); // 加入到PointToAdd中
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]); // 如果周围没有点或者没有初始化EKF，则加入到PointToAdd中
        }
    }

    // double start_time = omp_get_wtime();
    // //... vec1,vec2赋值
    // PointForLocalmap = PointToAdd;
    // PointForLocalmap.insert(PointForLocalmap.end(), PointNoNeedDownsample.begin(), PointNoNeedDownsample.end());

    // box_needrm_window_buffer.push_back(box_needrm);
    // // window incremental
    // feats_window_buffer.push_back(PointForLocalmap);
    // // std::cout << "PointToAddsize: " << PointToAdd.size() << " PointNoNeedDownsamplesize: " << PointNoNeedDownsample.size() << std::endl;
    // if (feats_window_buffer.size() > FEATS_WINDOW_SIZE)
    // {
    //     PointVector feats_needrm = feats_window_buffer.front();
    //     feats_window_buffer.pop_front();
    //     std::vector<BoxPointType> box_needrm = box_needrm_window_buffer.front();
    //     box_needrm_window_buffer.pop_front();
    //     // std::cout << "box_needrm size: " << box_needrm.size() << std::endl;
    //     ikdtree_submap.Delete_Point_Boxes(box_needrm);
    //     ikdtree_submap.Delete_Points(feats_needrm);
    // }

    // add_point_size = ikdtree_submap.Add_Points(PointToAdd, true);
    // ikdtree_submap.Add_Points(PointNoNeedDownsample, false);

    // // retrieve ikdtree submap
    // PointVector().swap(ikdtree_submap.PCL_Storage);
    // ikdtree_submap.flatten(ikdtree_submap.Root_Node, ikdtree_submap.PCL_Storage, NOT_RECORD);
    // featsSubMap->clear();
    // featsSubMap->points = ikdtree_submap.PCL_Storage;

    // double end_time = omp_get_wtime();
    // std::cout << "ikd-tree build submap time: " << end_time - start_time << std::endl;

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

/** the fuction where to add the rgb of point cloud **/

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{

    // PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_lidarbody); /** if dense_pub_enb true, use feats_undistort **/
    PointCloudXYZI::Ptr laserCloudFullRes(feats_down_submap_body);
    int size = laserCloudFullRes->points.size();

    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    // PointCloudXYZRGB::Ptr laserRgbCloudWorld(new PointCloudXYZRGB(size, 1));

    for (int i = 0; i < size; i++)
    {
        IntensitypointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]); // 修改后才是XYZI
        // RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserRgbCloudWorld->points[i]); // 转换出来是RGB，不是XYZI
    }
    //    static int scan_wait_num = 0;
    //    scan_wait_num ++;
    //    if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
    //    {
    //        pcd_index ++;
    //        string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
    //        pcl::PCDWriter pcd_writer;
    //        cout << "current scan saved to /PCD/" << all_points_dir << endl;
    //        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    //        pcl_wait_save->clear();
    //        scan_wait_num = 0;
    //    }

    // sensor_msgs::PointCloud2 laserCloudmsg;
    // pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    // laserCloudmsg.header.frame_id = odom_frame;
    // pubLaserCloudFull.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (0) // pcd_save_en
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));
        // PointCloudXYZRGB::Ptr laserRgbCloudWorld(new PointCloudXYZRGB(size, 1));

        for (int i = 0; i < size; i++)
        {
            IntensitypointBodyToWorld(&feats_undistort->points[i],
                                      &laserCloudWorld->points[i]);
            // RGBpointBodyToWorld(&feats_undistort->points[i],
            //                     &laserRgbCloudWorld->points[i]);
        }

        *pcl_wait_save += *laserCloudWorld;
        // *pcl_submap += *laserCloudWorld;

        // *pcl_wait_save_rgb += *laserRgbCloudWorld;
        // static int scan_wait_num = 0;
        // scan_wait_num++;
        // if (pcl_wait_save_rgb->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        // {
        //     pcd_index++;
        //     string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
        //     pcl::PCDWriter pcd_writer;
        //     cout << "current scan saved to /PCD/" << all_points_dir << endl;
        //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_rgb);
        //     pcl_wait_save_rgb->clear();
        //     scan_wait_num = 0;
        // }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_deskew_local(const ros::Publisher &pubLaserCloudDeskew)
{
    sensor_msgs::PointCloud2 laserCloudDeskew;
    pcl::toROSMsg(*feats_undistort, laserCloudDeskew);
    laserCloudDeskew.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudDeskew.header.frame_id = "body";
    pubLaserCloudDeskew.publish(laserCloudDeskew);
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        IntensitypointBodyToWorld(&laserCloudOri->points[i],
                                  &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = odom_frame;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odom_frame;
    pubLaserCloudMap.publish(laserCloudMap);
}

void publish_submap(const ros::Publisher &pubLaserSubMap)
{
    // generate submap
    PointCloudXYZI::Ptr laserCloudFullRes(feats_down_submap_body);
    int size = laserCloudFullRes->points.size();

    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    // PointCloudXYZRGB::Ptr laserRgbCloudWorld(new PointCloudXYZRGB(size, 1));

    for (int i = 0; i < size; i++)
    {
        IntensitypointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]); // 修改后才是XYZI
        // RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserRgbCloudWorld->points[i]); // 转换出来是RGB，不是XYZI
    }

    // 开始处理的时间
    // build submap using the sliding window
    pcl_submap.reset(new PointCloudXYZI());
    double start_time = omp_get_wtime();
    pcl_submap_vec.push_back(laserCloudWorld);
    // build submap using the sliding window
    if (pcl_submap_vec.size() > FEATS_WINDOW_SIZE)
    {
        // PointCloudXYZI::Ptr p = pcl_submap_vec.front();
        pcl_submap_vec.pop_front();
    }
    for (int i = 0; i < pcl_submap_vec.size(); i++)
    {
        *pcl_submap += *pcl_submap_vec[i];
    }
    std::cout << "pcl_submap_vec size: " << pcl_submap_vec.size() << " pcl_submap size: " << pcl_submap->size() << std::endl;
    double end_time = omp_get_wtime();
    std::cout << "incremental build submap time: " << end_time - start_time << std::endl;

    // publish submap
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*pcl_submap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odom_frame;
    pubLaserSubMap.publish(laserCloudMap);
}

void publish_infov(const ros::Publisher &pubLaserInFov)
{
    // generate
    PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
    int size = laserCloudFullRes->points.size();

    PointCloudXYZI::Ptr cloudInFov(new PointCloudXYZI(size, 1));
    
    // PointCloudXYZRGB::Ptr laserRgbCloudWorld(new PointCloudXYZRGB(size, 1));
    for (int i = 0; i < size; i++)
    {
        // fov seg
        
        double angle = atan2(laserCloudFullRes->points[i].y, laserCloudFullRes->points[i].x) * 180.0 / M_PI;
        if((angle> 0 && angle < FOV_DEG/2  )||(angle<0 && angle > -FOV_DEG/2))
        {
            IntensitypointBodyToWorld(&laserCloudFullRes->points[i], &cloudInFov->points[i]); // 修改后才是XYZI
        }
        
        // RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserRgbCloudWorld->points[i]); // 转换出来是RGB，不是XYZI
    }
    std::cout << "cloudInFov size: " << cloudInFov->size() << std::endl;
    *pcl_wait_save_infov += *cloudInFov;
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*cloudInFov, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odom_frame;
    pubLaserInFov.publish(laserCloudMap);
}

void publish_undistort(const ros::Publisher &pubLaserUndistort)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*feats_undistort, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odom_frame;
    pubLaserUndistort.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    Eigen::Quaterniond q(state_rot_world_body);
    out.pose.position.x = state_pos_world_body(0);
    out.pose.position.y = state_pos_world_body(1);
    out.pose.position.z = state_pos_world_body(2);
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    // cout << "-----publish_odometry-----" << endl;
    odomAftMapped.header.frame_id = odom_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    keyFrameOdom = odomAftMapped;
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    // write result to file
    if (lins_save_en)
    {
        foutLINS.setf(ios::fixed, ios::floatfield);
        foutLINS.precision(9);
        foutLINS << odomAftMapped.header.stamp.toSec() << " ";
        foutLINS.precision(6);
        foutLINS << state_pos_world_body(0) << " " << state_pos_world_body(1) << " " << state_pos_world_body(2) << " ";
        foutLINS << geoQuat.x << " " << geoQuat.y << " " << geoQuat.z << " " << geoQuat.w << endl;
        foutLINS.close();
    }

    // calc distance

    // if (pubKeyframePose.getNumSubscribers() > 0)
    // {
    //     nav_msgs::Odometry keyframePose;
    //     keyframePose.header = odomAftMapped.header;
    //     keyframePose.pose = odomAftMapped.pose;
    //     pubKeyframePose.publish(keyframePose);
    // }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, odom_frame, body_frame));
}

void publish_path(const ros::Publisher pubPath)
{
    // cout << "-----publish_path-----" << endl;
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = odom_frame;

    /*** if path is too large, the rviz will crash ***/
    //    static int jjj = 0;
    //    jjj++;
    //    if (jjj % 10 == 0)
    //    {
    path.poses.push_back(msg_body_pose);
    //        pubPath.publish(path);
    //    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_lidarbody->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_lidarbody->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/lins_save_en", lins_save_en, false);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("common/odom_frame", odom_frame, "odom");
    nh.param<string>("common/body_frame", body_frame, "lidar_link");
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<string>("common/cloud_deskewed_topic", cloud_deskewed_topic, "/cloud_deskewed");
    nh.param<string>("common/odometry_topic", odometry_topic, "/odometry");
    nh.param<string>("publish/lins_result_path", LINS_RESULT_PATH, "");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5); // 与feats_down_body有关
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);   // 与定位有关
    nh.param<double>("filter_size_submap", filter_size_submap_min, 0.01);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("deskew_enabled", deskew_enabled, true);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("mapping/lidar_to_body_T", lidar_to_body_T, vector<double>());
    nh.param<vector<double>>("mapping/lidar_to_body_R", lidar_to_body_R, vector<double>());
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    cout << "LINS_RESULT_PATH " << LINS_RESULT_PATH << endl;

    cout << "deskew_enabled " << deskew_enabled << endl;
    cout << "feature_extract_enable " << p_pre->feature_enabled << endl;
    cout << "point_filter_num " << p_pre->point_filter_num << endl;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = odom_frame;

    std::ofstream fout(LINS_RESULT_PATH, std::ios::out); // clear
    fout.close();

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min); // 这个是定位的时候用的
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);     // 好像没什么用
    downSizeFilterSubMap.setLeafSize(filter_size_submap_min, filter_size_submap_min, filter_size_submap_min);

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // 设置imu和lidar外参和imu参数等
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    Lidar_T_wrt_BODY << VEC_FROM_ARRAY(lidar_to_body_T);
    Lidar_R_wrt_BODY << MAT_FROM_ARRAY(lidar_to_body_R);
    IMU_T_wrt_BODY << -Lidar_R_wrt_BODY * Lidar_R_wrt_IMU.transpose() * Lidar_T_wrt_IMU + Lidar_T_wrt_BODY;
    IMU_R_wrt_BODY << Lidar_R_wrt_BODY * Lidar_R_wrt_IMU.transpose();
    BODY_T_wrt_IMU << -IMU_R_wrt_BODY.transpose() * IMU_T_wrt_BODY;
    BODY_R_wrt_IMU << IMU_R_wrt_BODY.transpose();
    std::cout << " IMU_T_wrt_BODY " << IMU_T_wrt_BODY.transpose() << std::endl;
    std::cout << " IMU_R_wrt_BODY " << std::endl
              << IMU_R_wrt_BODY << std::endl;
    std::cout << " BODY_T_wrt_IMU " << BODY_T_wrt_IMU.transpose() << std::endl;
    std::cout << " BODY_R_wrt_IMU " << std::endl
              << BODY_R_wrt_IMU << std::endl;
    // p_imu->set_deskew(deskew_enabled);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    foutLINS.open(LINS_RESULT_PATH, ios::out);
    foutLINS.close();
    foutLINS.open(LINS_RESULT_PATH, ios::app);

    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_img = it.subscribe("camera/image_raw", 200000, img_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_registered", 100000); /** active **/
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_effected", 100000);
    ros::Publisher pubLaserCloudDeskew = nh.advertise<sensor_msgs::PointCloud2>(cloud_deskewed_topic, 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("mapping/Laser_map", 100000);
    ros::Publisher pubLaserSubMap = nh.advertise<sensor_msgs::PointCloud2>("mapping/Local_map", 100000);
    ros::Publisher pubLaserInFov = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_infov", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("mapping/Odometry", 100000);
    ros::Publisher pubKeyframePose = nh.advertise<nav_msgs::Odometry>("mapping/keyframe_pose", 100000);

    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("mapping/path", 100000);
    //------------------------------------------------------------------------------------------------------
    static int count; /** count of the publish func **/
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort); /** get the feats_undistort points **/

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_lidarbody);
            downSizeFilterSubMap.setInputCloud(feats_undistort);
            downSizeFilterSubMap.filter(*feats_down_submap_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_lidarbody->points.size();
            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_lidarbody->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            getCurPose(state_point);
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            /******* Publish points *******/
            if (path_en)
            {
                publish_path(pubPath);
                count++;
                if (count % 10 == 0)
                {
                    pubPath.publish(path);
                    cout << "path_stamp:" << msg_body_pose.header.stamp << endl;
                    cout << "path_pose:" << path.poses.at(path.poses.size() - 1).pose.position.x << endl;
                    cout << "path_pose:" << path.poses[path.poses.size() - 1].pose.orientation.x << endl;
                    cout << "path_pose:" << msg_body_pose.pose.position.x << endl;
                    cout << "path_pose:" << msg_body_pose.pose.orientation.x << endl;
                }
            }
            if (scan_pub_en) // || pcd_save_en
            {
                // std::cout << " publish_frame_world " << std::endl;
                // publish_frame_world(pubLaserCloudFull); //   发布world系下的点云
                publish_submap(pubLaserSubMap);         //   发布局部地图
                publish_infov(pubLaserInFov);           //   发布视野内的点云
                // publish_undistort(pubLaserUndistort);   //   发布畸变纠正后的点云
            }

            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_deskew_local(pubLaserCloudDeskew);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/

    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    // {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     cout << "current scan saved to /PCD/" << file_name << endl;
    //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    // }

    // if (pcl_wait_save_rgb->size() > 0 && pcd_save_en)
    // {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     cout << "current scan saved to /PCD/" << file_name << endl;
    //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_rgb);
    // }

    
    string file_name = string("pcd_infov.pcd");
    string pcl_infov_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    cout << "final bin map saved to /PCD/" << file_name << endl;
    pcd_writer.writeBinary(pcl_infov_dir, *pcl_wait_save_infov);

    foutLINS.close();
    if (0)
    {
        // 最后保存一次global map
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;

        string file_name = string("map_bin.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "final bin map saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *featsFromMap);

        // file_name = string("map_ascii.pcd");
        // all_points_dir = (string(string(ROOT_DIR) + "PCD/") + file_name);
        // cout << "final txt map saved to /PCD/" << file_name << endl;
        // featsFromMap->height = 1;
        // featsFromMap->width = featsFromMap->size();
        // pcd_writer.writeASCII(all_points_dir, *featsFromMap);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
