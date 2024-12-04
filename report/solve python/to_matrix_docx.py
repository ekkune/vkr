'''while(True):
    a = []
    i = input()
    if i == "?":
        break
    else:
        b = i.split(" ")
        for k in b: 
            if k == '':
                continue
            else:
                a.append(k)
        st = 'left ('
        k = 0
        for i in range(len(a)):
            if i % 4 == 0:
                if k > 0:
                    st += "}"
                st += "stack{"
                k = 0
            st += str(a[i])
            if i % 4 != 3:
                st += "#"
            k += 1
        if k > 0:
            st += "}"
        st += "right )"
        print(st)'''
def format_list(lst):
    # Разбиваем список на подсписки по 4 элемента
    matrix = [lst[i:i+4] for i in range(0, len(lst), 4)]
    formatted_matrix = 'left ( ' + ' '.join([f'stack {{{" # ".join(map(str, col))}}}' for col in zip(*matrix)]) + ' right )'
    return formatted_matrix
lst = "    0.068966   0.344828   0.172414  -0.482759 0.250000  -0.125000   0.208333  -0.041667 -3.000000   5.500000   1.500000  -5.500000  0.307692   0.043956  -0.153846  -0.241758 ".split(" ")
lst_filtr = list(filter(None, lst))
print(lst_filtr)
#lst = [-2, -10, -5, 14, 12, -6, 10, -2, -6, 11, 3, -11, 14, 2, -7, -11]

formatted_list = format_list(lst_filtr)
print(formatted_list)
