operators3 = {'<<=', '>>=', '...'}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|=', '::'
    }
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}', '@', '!'
    }


def tokenize(line):
    line = line.replace('\\t', ' ')
    tmp, w = [], []
    i = 0
    while i < len(line):

        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1

        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1

        else:
            w.append(line[i])
            i += 1
    if len(w) > 0:
        tmp.append(''.join(w))

    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))


