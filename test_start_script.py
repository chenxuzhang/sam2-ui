from pathlib import Path


def test_start_script_uses_expected_conda_env_and_app():
    script = Path(__file__).with_name("start.sh").read_text()

    assert "CONDA_ENV=\"sam2-env\"" in script
    assert "APP_FILE=\"gradio_segment.py\"" in script
    assert "PORT=\"7860\"" in script
    assert "conda run -n \"${CONDA_ENV}\" python \"${APP_FILE}\"" in script


def test_start_script_kills_existing_port_processes():
    script = Path(__file__).with_name("start.sh").read_text()

    assert "lsof -ti tcp:${PORT}" in script
    assert "kill ${pids}" in script
    assert "kill -9 ${still_running}" in script
