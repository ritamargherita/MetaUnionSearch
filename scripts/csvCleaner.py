import os
import re
import sys

def clean_data(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        data = infile.read()
        
        #cleaned_data = re.sub(r'\\', '', data)
        #cleaned_data = re.sub(r';\n', ';', cleaned_data)
        cleaned_data = re.sub(r'" "', '', cleaned_data)
        cleaned_data = re.sub(r'</s>', '', cleaned_data)
        cleaned_data = re.sub(r';  ', ';', cleaned_data)
        cleaned_data = re.sub(r' ;', ';', cleaned_data)
        
        rows = cleaned_data.splitlines()
        header = rows[0]
        body = ' '.join(rows[1:])

        pattern = r'((\d+;)+)'
        parts = re.split(pattern, body)
        
        formatted_rows = [header]
        current_row = ''
        expected_index = 0

        for part in parts:
            if re.match(r'^(\d+;)+$', part): 
                numbers = part.split(';')[:-1]
                row_index = int(numbers[0])
                if row_index == expected_index:
                    if current_row:
                        formatted_rows.append(current_row.strip())
                    current_row = part
                    expected_index += 1
                else:
                    current_row += part
            else:
                current_row += part

        if current_row:
            formatted_rows.append(current_row.strip())

        return '\n'.join(formatted_rows)

def write_output_file(output_file_path, cleaned_data):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_data)

def process_file(input_file_path, output_file_path):
    cleaned_data = clean_data(input_file_path)
    write_output_file(output_file_path, cleaned_data)

def main(input_folder, output_folder):

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        process_file(input_file_path, output_file_path)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python csvCleaner.py <path_input_folder> <path_output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    main(input_folder, output_folder)

