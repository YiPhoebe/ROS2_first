from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Common args ----
    split          = DeclareLaunchArgument('split', default_value='train')
    frequency      = DeclareLaunchArgument('frequency', default_value='20.0')  # Hz
    loop           = DeclareLaunchArgument('loop', default_value='false')      # image_pub_test: loop over dataset
    img_root       = DeclareLaunchArgument('img_root', default_value='')       # custom dataset root; if set, overrides split default
    img_patterns   = DeclareLaunchArgument('img_patterns', default_value='**/*.jpg,**/*.jpeg,**/*.png')
    img_list_file  = DeclareLaunchArgument('img_list_file', default_value='')  # file with absolute image paths
    img_recursive  = DeclareLaunchArgument('img_recursive', default_value='true')
    # train은 셔플, val/test는 비셔플이 기본값이 되도록 동적 기본값 설정
    img_shuffle    = DeclareLaunchArgument(
        'img_shuffle',
        default_value=PythonExpression([
            "'true' if ",
            LaunchConfiguration('split'),
            " in ['train','TRAIN'] else 'false'",
        ]),
    )
    img_seed       = DeclareLaunchArgument('img_seed', default_value='42')

    # Detector A params
    weightsA       = DeclareLaunchArgument('weightsA', default_value='/workspace/yolo11n.pt')
    confA          = DeclareLaunchArgument('confA', default_value='0.25')
    imgszA         = DeclareLaunchArgument('imgszA', default_value='640')

    # Detector B params
    weightsB       = DeclareLaunchArgument('weightsB', default_value='/workspace/yolov8n.pt')
    confB          = DeclareLaunchArgument('confB', default_value='0.35')
    imgszB         = DeclareLaunchArgument('imgszB', default_value='416')

    # Overlay params
    overlay_conf   = DeclareLaunchArgument('overlay_conf', default_value='0.25')
    images_every_n = DeclareLaunchArgument('images_every_n', default_value='5')
    overlay_sync_ms= DeclareLaunchArgument('overlay_sync_ms', default_value='200')
    save_min_boxes = DeclareLaunchArgument('save_min_boxes', default_value='0')

    # Outputs (per split)
    frames_A       = DeclareLaunchArgument('frames_A', default_value='/workspace/pr_A/$(var split)/frames')
    frames_B       = DeclareLaunchArgument('frames_B', default_value='/workspace/pr_B/$(var split)/frames')
    out_csv_A      = DeclareLaunchArgument('out_csv_A', default_value='/workspace/pr_A/$(var split)/out_A.csv')
    out_csv_B      = DeclareLaunchArgument('out_csv_B', default_value='/workspace/pr_B/$(var split)/out_B.csv')
    out_json_A     = DeclareLaunchArgument('out_json_A', default_value='/workspace/pr_A/$(var split)/out_A.json')
    out_json_B     = DeclareLaunchArgument('out_json_B', default_value='/workspace/pr_B/$(var split)/out_B.json')
    out_mp4_A      = DeclareLaunchArgument('out_mp4_A', default_value='/workspace/pr_A/$(var split)/out_A.mp4')
    out_mp4_B      = DeclareLaunchArgument('out_mp4_B', default_value='/workspace/pr_B/$(var split)/out_B.mp4')

    # Image publisher from dataset (split=train/valid/test)
    img_pub = ExecuteProcess(
        cmd=[
            '/bin/bash', '-lc',
            "python3 /workspace/src/image_pub_test.py --ros-args "
            "-p split:='$(var split)' "
            "-p frequency:='$(var frequency)' "
            "-p loop:='$(var loop)' "
            "-p root:='$(var img_root)' "
            "-p patterns:='$(var img_patterns)' "
            "-p list_file:='$(var img_list_file)' "
            "-p recursive:='$(var img_recursive)' "
            "-p shuffle:='$(var img_shuffle)' "
            "-p seed:='$(var img_seed)'"
        ],
        output='screen',
    )

    # Overlay A (listens to /yoloA/bounding_boxes)
    overlay_A = ExecuteProcess(
        cmd=[
            '/bin/bash', '-lc',
            "python3 /workspace/src/overlay_viz.py --ros-args "
            "-p overlay_conf_min:='$(var overlay_conf)' "
            "-p save_images:=true -p images_dir:='$(var frames_A)' -p images_every_n:='$(var images_every_n)' "
            "-p sync_tol_ms:='$(var overlay_sync_ms)' -p save_min_boxes:='$(var save_min_boxes)' "
            "-p save_mp4:=true -p mp4_path:='$(var out_mp4_A)' "
            "-p save_boxes_csv:=true -p csv_path:='$(var out_csv_A)' "
            "-p save_boxes_json:=true -p json_path:='$(var out_json_A)' "
            "-p boxes_topic:=/yoloA/bounding_boxes "
            "-r /image_yolo:=/image_yolo_A -r __ns:=/A -r __node:=overlay_A"
        ],
        output='screen',
    )

    # Overlay B (listens to /yoloB/bounding_boxes)
    overlay_B = ExecuteProcess(
        cmd=[
            '/bin/bash', '-lc',
            "python3 /workspace/src/overlay_viz.py --ros-args "
            "-p overlay_conf_min:='$(var overlay_conf)' "
            "-p save_images:=true -p images_dir:='$(var frames_B)' -p images_every_n:='$(var images_every_n)' "
            "-p sync_tol_ms:='$(var overlay_sync_ms)' -p save_min_boxes:='$(var save_min_boxes)' "
            "-p save_mp4:=true -p mp4_path:='$(var out_mp4_B)' "
            "-p save_boxes_csv:=true -p csv_path:='$(var out_csv_B)' "
            "-p save_boxes_json:=true -p json_path:='$(var out_json_B)' "
            "-p boxes_topic:=/yoloB/bounding_boxes "
            "-r /image_yolo:=/image_yolo_B -r __ns:=/B -r __node:=overlay_B"
        ],
        output='screen',
    )

    # Detector A
    det_A = Node(
        package='yolo_subscriber_py',
        executable='yolo_subscriber_py_node',
        namespace='/A',
        name='yolo_sub_A',
        output='screen',
        parameters=[{
            'weights': ['$(var weightsA)'],
            'conf': ['$(var confA)'],
            'imgsz': ['$(var imgszA)'],
            'every_n': 1,
            'sub_topic': '/image_raw',
            'pub_topic': '/yoloA/bounding_boxes',
        }],
        # Determinism-related env for PyTorch/Ultralytics
        additional_env={
            'PYTHONHASHSEED': '42',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:2',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        },
    )

    # Detector B
    det_B = Node(
        package='yolo_subscriber_py',
        executable='yolo_subscriber_py_node',
        namespace='/B',
        name='yolo_sub_B',
        output='screen',
        parameters=[{
            'weights': ['$(var weightsB)'],
            'conf': ['$(var confB)'],
            'imgsz': ['$(var imgszB)'],
            'every_n': 1,
            'sub_topic': '/image_raw',
            'pub_topic': '/yoloB/bounding_boxes',
        }],
        additional_env={
            'PYTHONHASHSEED': '42',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:2',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        },
    )

    return LaunchDescription([
        # args
        split, frequency, loop, img_root, img_patterns, img_list_file, img_recursive, img_shuffle, img_seed,
        weightsA, confA, imgszA,
        weightsB, confB, imgszB,
        overlay_conf, images_every_n, overlay_sync_ms, save_min_boxes,
        frames_A, frames_B, out_csv_A, out_csv_B, out_json_A, out_json_B, out_mp4_A, out_mp4_B,
        # nodes
        img_pub,
        overlay_A,
        overlay_B,
        det_A,
        det_B,
    ])
