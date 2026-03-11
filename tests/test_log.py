from src.services.log_service import write_log
import os


def test_log_write():

    file = write_log("teste")

    assert os.path.exists(file)