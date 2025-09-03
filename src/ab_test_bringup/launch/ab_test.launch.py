from launch import LaunchDescription
from launch_ros.descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, OpaqueFunction
from launch_ros.actions import Node


def kv_param(name, value_subst):
    # Build a single '-p name:=value' argument with substitutions
    return ['-p', [TextSubstitution(text=f'{name}:='), value_subst]]


def generate_launch_description():
    # ---- args ----
    bag_path        = LaunchConfiguration('bag')
    rate            = LaunchConfiguration('rate')
    loop            = LaunchConfiguration('loop')
    start_paused    = LaunchConfiguration('start_paused')

    confA           = LaunchConfiguration('confA')
    imgszA          = LaunchConfiguration('imgszA')
    confB           = LaunchConfiguration('confB')
    imgszB          = LaunchConfiguration('imgszB')

    overlay_conf    = LaunchConfiguration('overlay_conf')
    images_every_n  = LaunchConfiguration('images_every_n')
    overlay_sync_ms = LaunchConfiguration('overlay_sync_ms')
    save_min_boxes  = LaunchConfiguration('save_min_boxes')

    frames_A        = LaunchConfiguration('frames_A')
    frames_B        = LaunchConfiguration('frames_B')
    out_csv_A       = LaunchConfiguration('out_csv_A')
    out_csv_B       = LaunchConfiguration('out_csv_B')
    out_json_A      = LaunchConfiguration('out_json_A')
    out_json_B      = LaunchConfiguration('out_json_B')
    out_mp4_A       = LaunchConfiguration('out_mp4_A')
    out_mp4_B       = LaunchConfiguration('out_mp4_B')

    # ---- overlay A/B 먼저 ----
    overlay_A_cmd = PythonExpression([
        "'python3 /workspace/src/overlay_viz.py --ros-args' +"
        " ' -p overlay_conf_min:=' + '", overlay_conf, "' +"
        " ' -p save_images:=true' +"
        " ' -p images_dir:=' + '", frames_A, "' +"
        " ' -p images_every_n:=' + '", images_every_n, "' +"
        " ' -p sync_tol_ms:=' + '", overlay_sync_ms, "' +"
        " ' -p save_min_boxes:=' + '", save_min_boxes, "' +"
        " ' -p save_mp4:=true -p mp4_path:=' + '", out_mp4_A, "' +"
        " ' -p save_boxes_csv:=true -p csv_path:=' + '", out_csv_A, "' +"
        " ' -p save_boxes_json:=true -p json_path:=' + '", out_json_A, "' +"
        " ' -p boxes_topic:=/yoloA/bounding_boxes' +"
        " ' -p use_sim_time:=true' +"
        " ' -r /image_yolo:=/image_yolo_A' +"
        " ' -r __ns:=/A' +"
        " ' -r __node:=overlay_A'"
    ])
    overlay_A = ExecuteProcess(cmd=['/bin/bash', '-lc', overlay_A_cmd], output='screen')

    overlay_B_cmd = PythonExpression([
        "'python3 /workspace/src/overlay_viz.py --ros-args' +"
        " ' -p overlay_conf_min:=' + '", overlay_conf, "' +"
        " ' -p save_images:=true' +"
        " ' -p images_dir:=' + '", frames_B, "' +"
        " ' -p images_every_n:=' + '", images_every_n, "' +"
        " ' -p sync_tol_ms:=' + '", overlay_sync_ms, "' +"
        " ' -p save_min_boxes:=' + '", save_min_boxes, "' +"
        " ' -p save_mp4:=true -p mp4_path:=' + '", out_mp4_B, "' +"
        " ' -p save_boxes_csv:=true -p csv_path:=' + '", out_csv_B, "' +"
        " ' -p save_boxes_json:=true -p json_path:=' + '", out_json_B, "' +"
        " ' -p boxes_topic:=/yoloB/bounding_boxes' +"
        " ' -p use_sim_time:=true' +"
        " ' -r /image_yolo:=/image_yolo_B' +"
        " ' -r __ns:=/B' +"
        " ' -r __node:=overlay_B'"
    ])
    overlay_B = ExecuteProcess(cmd=['/bin/bash', '-lc', overlay_B_cmd], output='screen')

    # ---- detector A/B 그 다음 ----
    det_A = Node(
        package='yolo_subscriber_py',
        executable='yolo_subscriber_py_node',
        namespace='/A',
        name='yolo_sub_A',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'conf':  ParameterValue(confA, value_type=float),
            'imgsz': ParameterValue(imgszA, value_type=int),
            'every_n': 1,
            'pub_topic': '/yoloA/bounding_boxes',
        }],
        remappings=[
            ('/image_raw', '/image_raw'),
            ('/yolo/bounding_boxes', '/yoloA/bounding_boxes'),
            ('/image_yolo', '/image_yolo_A'),
        ]
    )

    det_B = Node(
        package='yolo_subscriber_py',
        executable='yolo_subscriber_py_node',
        namespace='/B',
        name='yolo_sub_B',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'conf': confB,
            'imgsz': imgszB,
            'every_n': 1,
            'pub_topic': '/yoloB/bounding_boxes',
        }],
        remappings=[
            ('/image_raw', '/image_raw'),
            ('/yolo/bounding_boxes', '/yoloB/bounding_boxes'),
            ('/image_yolo', '/image_yolo_B'),
        ]
    )

    # ---- bag 재생: overlay & detector가 올라온 뒤 시작 ----
    # OpaqueFunction으로 LaunchConfiguration 값을 평가해 조건부 플래그를 구성
    def _make_bag_proc(context):
        bag = LaunchConfiguration('bag').perform(context)
        rate_val = LaunchConfiguration('rate').perform(context)
        loop_val = LaunchConfiguration('loop').perform(context).lower()
        pause_val = LaunchConfiguration('start_paused').perform(context).lower()

        cmd = [
            'ros2', 'bag', 'play', bag,
            '--clock',
            '--read-ahead-queue-size', '10',
            '--rate', rate_val,
        ]
        if loop_val == 'true':
            cmd.append('--loop')
        if pause_val == 'true':
            cmd.append('--start-paused')
        return [ExecuteProcess(cmd=cmd, output='screen')]

    # Timer로 순서 제어(overlay/detector 올라간 뒤 bag 시작)
    start_bag_after = TimerAction(period=2.0, actions=[OpaqueFunction(function=_make_bag_proc)])

    return LaunchDescription([
        # ---- declare args ----
        DeclareLaunchArgument('bag',           default_value='/workspace/bags/argo_full/argo_full_0.mcap'),
        DeclareLaunchArgument('rate',          default_value='0.8'),
        DeclareLaunchArgument('loop',          default_value='false'),        # 'true' / 'false'
        DeclareLaunchArgument('start_paused',  default_value='false'),        # 'true' / 'false'

        DeclareLaunchArgument('confA',  default_value='0.25'),
        DeclareLaunchArgument('imgszA', default_value='640'),
        DeclareLaunchArgument('confB',  default_value='0.35'),
        DeclareLaunchArgument('imgszB', default_value='416'),

        DeclareLaunchArgument('overlay_conf',   default_value='0.25'),
        DeclareLaunchArgument('images_every_n', default_value='5'),
        DeclareLaunchArgument('overlay_sync_ms', default_value='200'),
        DeclareLaunchArgument('save_min_boxes', default_value='0'),

        DeclareLaunchArgument('frames_A',  default_value='/workspace/frames_A'),
        DeclareLaunchArgument('frames_B',  default_value='/workspace/frames_B'),
        DeclareLaunchArgument('out_csv_A', default_value='/workspace/out_A.csv'),
        DeclareLaunchArgument('out_csv_B', default_value='/workspace/out_B.csv'),
        DeclareLaunchArgument('out_json_A', default_value='/workspace/out_A.json'),
        DeclareLaunchArgument('out_json_B', default_value='/workspace/out_B.json'),
        DeclareLaunchArgument('out_mp4_A', default_value='/workspace/out_A.mp4'),
        DeclareLaunchArgument('out_mp4_B', default_value='/workspace/out_B.mp4'),

        # ---- run order: overlay -> detector -> (delay) bag ----
        overlay_A,
        overlay_B,
        det_A,
        det_B,
        start_bag_after,
    ])
