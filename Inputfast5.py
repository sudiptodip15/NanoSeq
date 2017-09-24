import poretools

class Data(object):

    @classmethod
    def get_data(cls,path_to_data):
      with  open(path_to_data,'r') as file_handler :
        read_data_obj = poretools.Fast5File(path_to_data)
        read_data = read_data_obj.get_template_events()
      return read_data

if __name__=='__main__':
    path_to_data="Trialdata/example_1.fast5"
    read_data = Data.get_data(path_to_data)
    fout = open('Event_example_1.txt','w')
    for rd in read_data:
        fout.write(str(rd)+"\n")
    fout.close()
