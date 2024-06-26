class SCPFileObject :
    # 4 fields that would be passed
    def __init__(self, image_arr, action, username, params) :
        self.image_arr = image_arr
        self.action = action
        self.username = username
        self.params = params

    def set_scp_fields(self, image_arr, action, username, params) :
        self.image_arr = image_arr
        self.action = action
        self.username = username
        self.params = params

    def __str__(self) :
        return f"SCPFileObject : action = {self.action}, image_Arr = {self.image_arr}, username = {self.username}, params = {self.params}"