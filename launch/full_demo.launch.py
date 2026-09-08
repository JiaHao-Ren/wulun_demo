# 语音对话和表情识别一起起。
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _espeak_paths():
    try:
        import piper_phonemize
        pkg = os.path.dirname(piper_phonemize.__file__)
        data = os.path.join(pkg, 'espeak-ng-data')
        libs = os.path.join(os.path.dirname(pkg), 'piper_phonemize.libs')
        so = ''
        if os.path.isdir(libs):
            for f in sorted(os.listdir(libs)):
                if f.startswith('libespeak-ng') and '.so' in f:
                    so = os.path.join(libs, f)
                    break
        return so, data
    except Exception:
        return '', ''


def _haarcascade():
    # OpenCV 自带的人脸检测 xml
    prefix = os.environ.get('CONDA_PREFIX', '/usr')
    for rel in ('share/opencv4/haarcascades', 'share/OpenCV/haarcascades'):
        p = os.path.join(prefix, rel, 'haarcascade_frontalface_default.xml')
        if os.path.exists(p):
            return p
    return ''


def launch_setup(context, *args, **kwargs):
    models = LaunchConfiguration('models_dir').perform(context)
    source = LaunchConfiguration('source').perform(context)
    camera = LaunchConfiguration('camera').perform(context)
    backend = LaunchConfiguration('backend').perform(context)
    pkg_share = get_package_share_directory('edge_inference_optimizer')

    espeak_lib, espeak_data = _espeak_paths()
    if not espeak_lib:
        raise RuntimeError('libespeak-ng not found:  pip install piper-phonemize')
    cascade = _haarcascade()
    if not cascade:
        raise RuntimeError('haarcascade_frontalface_default.xml not found in $CONDA_PREFIX')

    nodes = []

    nodes.append(Node(
        package='edge_inference_optimizer', executable='audio_capture_node',
        name='audio_capture', output='screen',
        parameters=[{
            'source': source,
            'wav_path': LaunchConfiguration('wav_path'),
            'loop': LaunchConfiguration('loop'),
            'mic_sample_rate': 16000, 'chunk_samples': 512,
            'lead_in_silence_ms': 300, 'tail_silence_ms': 1500,
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='vad_node',
        name='vad_node', output='screen',
        parameters=[{
            'model_path': os.path.join(models, 'silero_vad.onnx'),
            'min_silence_ms': LaunchConfiguration('min_silence_ms'),
            'pre_roll_ms': 300.0, 'min_speech_ms': 250.0, 'threads': 1,
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='asr_node',
        name='asr_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'whisper_base_onnx'),
            'language': 'zh', 'simplified_chinese': True,
            'backend': backend, 'threads': 6, 'max_tokens': 60,
            'endpoint_wait_ms': LaunchConfiguration('min_silence_ms'),
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='llm_node',
        name='llm_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'qwen05b_onnx'),
            'backend': backend, 'threads': 8,
            'max_new_tokens': LaunchConfiguration('max_new_tokens'),
            'history_turns': 3,
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='tts_node',
        name='tts_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'piper_zh'),
            'espeak_lib': espeak_lib, 'espeak_data': espeak_data,
            'voice': 'cmn', 'threads': 4,
            'length_scale': LaunchConfiguration('length_scale'),
            'dump_dir': LaunchConfiguration('dump_dir'),
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='audio_playback_node',
        name='audio_playback', output='screen',
        condition=IfCondition(LaunchConfiguration('play_audio'))))

    cam_params = {'fps': LaunchConfiguration('fps'), 'width': 640, 'height': 480}
    if camera == 'device':
        cam_params['device_id'] = LaunchConfiguration('device_id')
    else:
        cam_params['video_path'] = LaunchConfiguration('video_path')
    nodes.append(Node(
        package='edge_inference_optimizer', executable='camera_node',
        name='camera_node', output='screen', parameters=[cam_params]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='emotion_detector_node',
        name='emotion_detector', output='screen',
        parameters=[{
            'model_path': os.path.join(models, 'emotion-ferplus-8.onnx'),
            'face_cascade': cascade, 'backend': backend, 'threads': 4,
        }]))
    nodes.append(Node(
        package='edge_inference_optimizer', executable='servo_mapper_node',
        name='servo_mapper', output='screen',
        parameters=[os.path.join(pkg_share, 'config', 'servo_mapping.yaml')]))

    nodes.append(Node(
        package='edge_inference_optimizer', executable='latency_reporter_node',
        name='latency_reporter', output='screen',
        parameters=[{
            'report_period_s': LaunchConfiguration('report_period_s'),
            'min_samples': 2,
        }]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('models_dir'),
        DeclareLaunchArgument('source', default_value='file', description='mic | file'),
        DeclareLaunchArgument('camera', default_value='file', description='device | file'),
        DeclareLaunchArgument('wav_path', default_value=''),
        DeclareLaunchArgument('video_path', default_value=''),
        DeclareLaunchArgument('device_id', default_value='0'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('backend', default_value='cpu', description='cpu | cuda'),
        DeclareLaunchArgument('min_silence_ms', default_value='500.0'),
        DeclareLaunchArgument('max_new_tokens', default_value='36'),
        DeclareLaunchArgument('length_scale', default_value='1.0'),
        DeclareLaunchArgument('fps', default_value='15.0'),
        DeclareLaunchArgument('play_audio', default_value='true'),
        DeclareLaunchArgument('dump_dir', default_value=''),
        DeclareLaunchArgument('report_period_s', default_value='25.0'),
        OpaqueFunction(function=launch_setup),
    ])
