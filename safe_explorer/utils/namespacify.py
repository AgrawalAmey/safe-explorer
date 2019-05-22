

class Namespacify(object):
  def __init__(self, in_dict):
    for key in in_dict.keys():
        if isinstance(in_dict[key], dict):
            in_dict[key] = Namespacify(in_dict[key])
    self.__dict__.update(in_dict)