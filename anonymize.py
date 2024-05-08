import os
import re
import shutil
keywords = frozenset({'static', 'case', 'long', 'class', 'interface', 'while', 'transient', 'null', 'package', 'throw',
                      'for', 'do', 'const', 'this', 'false', 'true', 'volatile', 'continue', 'final', 'instanceof',
                      'goto', 'double', 'int', 'void', 'throws', 'try', 'default', 'short', 'byte', 'private', 'break',
                      'assert', 'return', 'import', 'new', 'enum', 'strictfp', 'catch', 'char', 'super', 'implements',
                      'extends', 'switch', 'boolean', 'native', 'protected', 'float', 'synchronized', 'else', 'public',
                      'abstract', 'finally', 'if'})
main_fun = frozenset({'main', 'init', 'equals', 'finalize', 'contains', 'length', 'concat', 'replace', 'trim', 'append',
                      'insert', 'delete', 'reverse', 'exit', 'println', 'print', 'run', 'start', 'stop', 'sleep',
                      'remove', 'add', 'put', 'get', 'next', 'read', 'write', 'open', 'close', 'sort', 'clone', 'copy'})
main_cls = frozenset({'System', 'String', 'Object', 'Map', 'List', 'Date', 'Calendar', 'Set', 'Math', 'Arrays',
                      'Collections', 'Process', 'Random', 'File', 'Byte', 'Short', 'Integer', 'Long', 'Float', 'Double',
                      'Character', 'Boolean', 'Date', 'Thread'})


def clean_gadget(gadget, fun_symbols, var_symbols, str_symbols, cls_symbols):
    pattern = r'(?<!\S)/\*[\s\S]*?\*/'

    str_gadget = ''.join(gadget)
    while True:
        match = re.search(pattern, str_gadget)
        if match:
            match_list = [match.group()] if match else []
            for match in match_list:
                repl = '\n' * (match.count('\n') - 1)
                str_gadget = str_gadget.replace(match, repl)
        else:
            break

    gadget = str_gadget.split('\n')
    gadget = [line + '\n' for line in gadget]

    fun_count = len(fun_symbols.keys())
    var_count = len(var_symbols.keys())
    str_count = len(str_symbols.keys())
    cls_count = len(cls_symbols.keys())

    rx_all = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')
    rx_str = re.compile(r'"(?:\\.|[^"\\])*"')

    clean_gadget = []
    for line in gadget:
        line1 = re.sub(r'[^\x00-\x7f]', r'', line)

        user_str = re.findall(rx_str, line)
        for str_name in user_str:
            if str_name not in str_symbols.keys():
                str_symbols[str_name] = 's' + str(str_count)
                str_count += 1
            str_name1 = str_name[1:-1]
            str_name1 = re.escape(str_name1)
            line1 = re.sub(rf'"{str_name1}"', '"' + str_symbols[str_name] + '"', line1)

        user_all = rx_all.findall(line1)
        user_fun = rx_fun.findall(line1)
        user_var = rx_var.findall(line1)

        for word in user_all:
            if word in keywords or word in main_fun or word in main_cls:
                continue

            if len(word) == 1:
                continue
            try:
                index = line1.index(word)
            except:
                continue

            if word in str_symbols.values():
                if index > 0 and line1[index - 1] == '"':
                    continue
            if index > 0 and line1[index - 1] == "'":
                continue

            if word[0].isupper():
                if word not in cls_symbols.keys():
                    cls_symbols[word] = 'c' + str(cls_count)
                    cls_count += 1
                line1 = re.sub(r'\b(' + word + r')\b', cls_symbols[word], line1)
            else:
                if word in user_fun:
                    if word not in fun_symbols.keys():
                        fun_symbols[word] = 'f' + str(fun_count)
                        fun_count += 1
                    line1 = re.sub(r'\b(' + word + r')\b(?=\s*\()', fun_symbols[word], line1)
                if word in user_var:
                    if word not in var_symbols.keys():
                        var_symbols[word] = 'v' + str(var_count)
                        var_count += 1
                    line1 = re.sub(r'\b(' + word + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', var_symbols[word], line1)
        if line1.strip() == '':
            continue
        clean_gadget.append(line1)
    return clean_gadget, fun_symbols, var_symbols, str_symbols, cls_symbols


def main():
    root = 'data/'
    outRoot = 'positive/'
    for commit in os.listdir(root):
        if os.path.exists(outRoot + commit + '/outA'):
            continue
        if os.path.exists(outRoot + commit):
            shutil.rmtree(outRoot + commit)
        fun_symbols = {}
        var_symbols = {}
        str_symbols = {}
        cls_symbols = {}
        for file in os.listdir(root + commit + '/a/'):
            if 'test' in file or 'Test' in file:
                continue
            filePath = root + commit + '/a/' + file
            outPath = outRoot + commit + '/a/'
            f = open(filePath, 'r', encoding='utf-8')
            gadget = f.readlines()
            f.close()
            flag = False
            for line in gadget:
                if line.strip() != '':
                    if line.startswith('<!DOCTYPE html>'):
                        flag = True
                    break
            if flag:
                continue

            result, fun_symbols, var_symbols, str_symbols, cls_symbols = clean_gadget(gadget, fun_symbols, var_symbols, str_symbols, cls_symbols)

            if not os.path.exists(outPath):
                os.makedirs(outPath)
            f = open(outPath + file, 'w+', encoding='utf-8')
            f.writelines(result)
            f.close()
        for file in os.listdir(root + commit + '/b/'):
            if 'test' in file or 'Test' in file:
                continue
            filePath = root + commit + '/b/' + file
            outPath = outRoot + commit + '/b/'
            f = open(filePath, 'r', encoding='utf-8')
            gadget = f.readlines()
            f.close()
            flag = False
            for line in gadget:
                if line.strip() != '':
                    if line.startswith('<!DOCTYPE html>'):
                        flag = True
                    break
            if flag:
                continue
            result, fun_symbols, var_symbols, str_symbols, cls_symbols = clean_gadget(gadget, fun_symbols, var_symbols, str_symbols, cls_symbols)

            if not os.path.exists(outPath):
                os.makedirs(outPath)
            f = open(outPath + file, 'w+', encoding='utf-8')
            f.writelines(result)
            f.close()
        print(commit)



if __name__ == '__main__':
    main()
