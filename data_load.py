def load_int_file(filename):
    lines = []
    for line in file(filename):
        items = map(int, line.strip().split(' '))
        lines.append(items)
    return lines

def load_float_file(filename):
    lines = []
    for line in file(filename):
        if len(line.strip()) == 0:
            lines.append([])
            continue
        items = map(float, line.strip().split('\t'))
        lines.append(items)
    return lines

