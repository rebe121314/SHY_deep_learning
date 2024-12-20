import json
import dropbox
from io import BytesIO

# Dropbox access token
DROPBOX_ACCESS_TOKEN = 'sl.B6KMkulNKnGk6-6GiYBR0SUy_4M9JUjj2a_6kHeEwb_xz1KjMh-3TpzzO-rjU8CfDPpsP8hAVodmtr3901KHOFV5robf0_wZC-yxbt7sgk1lk1LwPZGIUjliYZrcQNbaL82dbrGVJEc5'
# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def process_file(file_content, filename, dropbox_output_folder):
    data = json.loads(file_content)
    output_list = []

    for shape in data['shapes']:
        label = shape['label']
        if label in ["w_GB", "s_GB"]:
            print("yeah!")
            points = shape['points']
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]

            bounding_box = [
                min(x_coordinates),
                min(y_coordinates),
                max(x_coordinates),
                max(y_coordinates)
            ]

            cell_id = data.get('version', 0.0)  # Adjust this line if you have a different way to get Cell ID
            granzyme_b = 0.0  # Replace with actual Granzyme B value if available
            x_position = int((bounding_box[0] + bounding_box[2]) / 2)
            y_position = int((bounding_box[1] + bounding_box[3]) / 2)

            output_data = {
                "Cell ID": cell_id,
                "X Position": x_position,
                "Y Position": y_position,
                "Bounding Box": bounding_box,
                "Granzyme B": granzyme_b
            }

            output_list.append(output_data)

        else:
          print("nope")

    if output_list:
        output_filename = filename.replace('_Granzyme B_path_view', '_labels')
        output_content = json.dumps(output_list, indent=4).encode('utf-8')

        dropbox_output_path = f"{dropbox_output_folder}/{output_filename}".replace('//', '/')
        dbx.files_upload(output_content, dropbox_output_path, mode=dropbox.files.WriteMode('overwrite'))

def extract_bounding_boxes(dropbox_annotation_folder, dropbox_output_folder):
    has_more = True
    cursor = None

    while has_more:
        if cursor:
            result = dbx.files_list_folder_continue(cursor)
        else:
            result = dbx.files_list_folder(dropbox_annotation_folder)

        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(".json"):
                _, res = dbx.files_download(path=entry.path_lower)
                file_content = res.content.decode('utf-8')
                process_file(file_content, entry.name, dropbox_output_folder)

        has_more = result.has_more
        cursor = result.cursor

# Example usage
dropbox_annotation_folder = '/Lables/testing'
dropbox_output_folder = '/Lables/manual_box_label'
extract_bounding_boxes(dropbox_annotation_folder, dropbox_output_folder)