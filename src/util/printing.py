OUT_WIDTH = 80

def print_header(header):
    prl = (OUT_WIDTH//2-len(header)//2) - 1
    prr = OUT_WIDTH - prl - len(header) - 2
    print("#" + " "*prl + header + " "*prr + "#")
    return


def print_border():
    print("-"*OUT_WIDTH)
    return
