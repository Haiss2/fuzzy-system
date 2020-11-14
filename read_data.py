from csv import reader
def int_or_float(x):
    try:
        return int(x)
    except:
        return float(x)

def read_file(path, mode='str'):
    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)

        if mode == 'float':
            list_of_rows = [ [int_or_float(j) for j in i] for i in list_of_rows]
        if mode == 'int':
            list_of_rows = [ [int(j) for j in i] for i in list_of_rows]

        return list_of_rows
