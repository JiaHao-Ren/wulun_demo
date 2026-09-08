# 只跑语音对话。
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _espeak_paths():
    # espeak 在 piper 包里，启动时再找路径。
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


def launch_setup(context, *args, **kwargs):
    models = LaunchConfiguration('models_dir').perform(context)
    source = LaunchConfiguration('source').perform(context)
    backend = LaunchConfiguration('backend').perform(context)
    espeak_lib, espeak_data = _espeak_paths()

    if not espeak_lib or not os.path.exists(espeak_lib):
        raise RuntimeError(
            'libespeak-ng not found. Install it with:  pip install piper-phonemize')

    capture = Node(
        package='wulun_demo', executable='audio_capture_node',
        name='audio_capture', output='screen',
        parameters=[{
            'source': source,
            'wav_path': LaunchConfiguration('wav_path'),
            'loop': LaunchConfiguration('loop'),
            'mic_sample_rate': 16000,
            'chunk_samples': 512,
            'lead_in_silence_ms': 300,
            'tail_silence_ms': 1500,
        }],
    )

    vad = Node(
        package='wulun_demo', executable='vad_node',
        name='vad_node', output='screen',
        parameters=[{
            'model_path': os.path.join(models, 'silero_vad.onnx'),
            'min_silence_ms': LaunchConfiguration('min_silence_ms'),
            'pre_roll_ms': 300.0,
            'min_speech_ms': 250.0,
            'threads': 1,
        }],
    )

    asr = Node(
        package='wulun_demo', executable='asr_node',
        name='asr_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'whisper_base_onnx'),
            'language': LaunchConfiguration('language'),
            'simplified_chinese': True,
            'backend': backend,
            'threads': 6,
            'max_tokens': 60,
            'endpoint_wait_ms': LaunchConfiguration('min_silence_ms'),
        }],
    )

    llm = Node(
        package='wulun_demo', executable='llm_node',
        name='llm_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'qwen05b_onnx'),
            'backend': backend,
            'threads': 8,
            'max_new_tokens': LaunchConfiguration('max_new_tokens'),
            'history_turns': 3,
        }],
    )

    tts = Node(
        package='wulun_demo', executable='tts_node',
        name='tts_node', output='screen',
        parameters=[{
            'model_dir': os.path.join(models, 'piper_zh'),
            'espeak_lib': espeak_lib,
            'espeak_data': espeak_data,
            'voice': 'cmn',
            'threads': 4,
            'length_scale': LaunchConfiguration('length_scale'),
            'dump_dir': LaunchConfiguration('dump_dir'),
        }],
    )

    playback = Node(
        package='wulun_demo', executable='audio_playback_node',
        name='audio_playback', output='screen',
        condition=IfCondition(LaunchConfiguration('play_audio')),
    )

    reporter = Node(
        package='wulun_demo', executable='latency_reporter_node',
        name='latency_reporter', output='screen',
        parameters=[{
            'report_period_s': LaunchConfiguration('report_period_s'),
            'min_samples': 2,
        }],
    )

    return [capture, vad, asr, llm, tts, playback, reporter]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('models_dir'),
        DeclareLaunchArgument('source', default_value='file'),
        DeclareLaunchArgument('wav_path', default_value=''),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('language', default_value='zh'),
        DeclareLaunchArgument('backend', default_value='cpu'),
        DeclareLaunchArgument('min_silence_ms', default_value='700.0'),
        DeclareLaunchArgument('max_new_tokens', default_value='36'),
        DeclareLaunchArgument('length_scale', default_value='1.0'),
        DeclareLaunchArgument('play_audio', default_value='true'),
        DeclareLaunchArgument('dump_dir', default_value=''),
        DeclareLaunchArgument('report_period_s', default_value='20.0'),
        OpaqueFunction(function=launch_setup),
    ])
