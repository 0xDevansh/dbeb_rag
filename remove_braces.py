import os
import re

current_directory = '.' 

pattern = re.compile(r"test_context \((\d+)\)\.pdf")

print("Starting file renaming...")

for filename in os.listdir(current_directory):
    match = pattern.match(filename)
    if match:
        number = match.group(1)
        
        new_filename = f"test_context_{number}.pdf"
        
        old_path = os.path.join(current_directory, filename)
        new_path = os.path.join(current_directory, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed: '{filename}'  ->  '{new_filename}'")

print("Done!")