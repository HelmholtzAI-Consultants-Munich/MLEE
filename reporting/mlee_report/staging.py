import os
import shutil
import subprocess
from auxiliary.turbopath import turbopath


def _chownfix(input_folder: str):
    # Windows does not support os.getuid() or chown, so nothing to do.
    if os.name == "nt":
        return

    # On Linux/macOS, fix ownership (mainly for Docker-created files)
    # TODO when running in docker think about supplying this UID
    current_uid = str(os.getuid())
    rightCorrection = "chown", "-R", current_uid, input_folder
    subprocess.run(rightCorrection)


# def makeSessionSubDir(sessionFolder: str, subDirName: str):
#     subDir = os.path.normpath(sessionFolder + "/" + subDirName + "/")
#     os.makedirs(subDir)
#     _chownfix(inputFolder=sessionFolder)
#     return subDir


def _clean_stage(input_folder: str):
    _chownfix(input_folder)

    print("cleaning up before export")
    # cleanup
    file_list = os.listdir(input_folder)
    try:
        file_list.remove(".dockerignore")
    except Exception as e:
        print(".dockerignore not present")
    try:
        file_list.remove(".gitignore")
    except Exception as e:
        print(".gitignore not present")

    for the_file in file_list:
        file_path = os.path.join(input_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print("finished cleaning - ready for export!")


def create_stage(
    session_dir: str,
    stage_dir: str = "the_stage",
):
    session_dir = turbopath(session_dir)
    stage_dir = turbopath(stage_dir)

    _clean_stage(stage_dir)

    # copy the template files
    shutil.copytree(
        "reporting/mlee_report/template",
        stage_dir,
        dirs_exist_ok=True,
    )

    # copy report contents
    shutil.copytree(session_dir, stage_dir + "/session")
    # TODO other contents


def move_from_stage(
    stage_dir: str,
    output_dir: str,
    overwrite: bool = True,
):
    stage_dir = turbopath(stage_dir)
    output_dir = turbopath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if overwrite == True:
        _clean_stage(input_folder=output_dir)

    shutil.copytree(stage_dir + "/resources", output_dir + "/resources")
    shutil.copytree(stage_dir + "/session", output_dir + "/session")
    shutil.copyfile(stage_dir + "/mlee.html", output_dir + "/mlee.html")
