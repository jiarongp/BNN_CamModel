def write_log(log_file, msg):
    with open(log_file, 'a') as f:
        f.write(msg)
        print(msg)