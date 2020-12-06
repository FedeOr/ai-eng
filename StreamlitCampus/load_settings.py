
import json

def get_setting(key):
    
    with open("local.settings.json", "r") as read_file:
      settings = json.load(read_file)
      
    value = settings.get(key)
    
    return value