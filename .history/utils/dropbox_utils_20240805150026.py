# dropbox_utils.py
# utils/dropbox_utils.py
import dropbox
import io
import python_calamine
import pandas as pd
from config import DROPBOX_ACCESS_TOKEN

dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def list_files_in_folder(folder_path):
    files = []
    result = dbx.files_list_folder(folder_path)
    files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        files.extend([entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)])
    return files

def read_image_from_dropbox(dropbox_path):
    _, res = dbx.files_download(path=dropbox_path)
    file_bytes = io.BytesIO(res.content)
    image = skio.imread(file_bytes)
    return image

def read_excel_from_dropbox(dropbox_path):
    _, res = dbx.files_download(path=dropbox_path)
    file_bytes = io.BytesIO(res.content)
    rows = iter(python_calamine.CalamineWorkbook.from_filelike(file_bytes).get_sheet_by_index(0).to_python())
    headers = list(map(str, next(rows)))
    data = [dict(zip(headers, row)) for row in rows]
    return pd.DataFrame(data)
