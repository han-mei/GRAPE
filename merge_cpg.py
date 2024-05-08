import os, re
import sys
import shutil

import javalang
import numpy as np

class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def importCPG(path):
    nodesByFunc, edgesByFunc = {}, {}
    listdirs = os.listdir(path)
    func_file = {}
    for dir in listdirs:
        cpgPath = path + '/' + dir + '/'
        files = os.listdir(cpgPath)
        for f in files:
            with open(cpgPath + f, 'r') as file:
                lines = file.readlines()
            if '<SUB>' not in lines[1] and '&lt;clinit&gt;' not in lines[1]: continue
            line1 = lines[0].strip()
            funcName = re.findall(r'"(.*?)"', line1)[0]
            funcName = replace_html_entities(funcName)
            funcName = dir + '_' + funcName
            nodes, edges = [], []
            pattern = r'^"(\d+)" -> "(\d+)"'
            for line in (lines[i].strip() for i in range(1, len(lines) - 1)):
                line = replace_html_entities(line)
                if re.match(pattern, line):
                    edges.append(line)
                else:
                    nodes.append(line)
            func_file[funcName] = dir
            nodesByFunc[funcName] = nodes
            edgesByFunc[funcName] = edges
    return nodesByFunc, edgesByFunc, func_file


def replace_html_entities(text):
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", "\"")
    text = text.replace("&apos;", "'")
    text = text.replace("&nbsp;", " ")
    text = text.replace("\\\012", "")
    return text


def nodeAttr(node):
    pattern1 = r'"(.*?)"'
    pattern2 = r'<.*'
    n1 = re.findall(pattern1, node)[0]
    n2 = re.findall(pattern2, node)[0][1:-3]
    return n1, n2


def edgeAttr(edge):
    pattern = r'"(.*?)"'
    e = re.findall(pattern, edge)
    if e[2][:3] == 'AST' or e[2][:3] == 'CDG' or e[2][:3] == 'CFG':
        e[2] = e[2][:3]
    return e[0], e[1], e[2]


def ifSameLine(n1, n2):
    if '<SUB>' not in n1[1] or '</SUB>' not in n2[1]:
        return False
    i = n1[1].index('<SUB>')
    j = n1[1].index('</SUB>')
    line1 = n1[1][i + 5:j]
    i = n2[1].index('<SUB>')
    j = n2[1].index('</SUB>')
    line2 = n2[1][i + 5:j]
    if line1 == line2:
        return True
    return False


def isAlphaAllUpper(s):
    for i in range(len(s)):
        if s[i].isalpha() and s[i].islower():
            return False
    return True


def codeCompletion(path, line_num):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    lineStart = line_num
    for i in range(line_num - 1, 0, -1):
        thisLine = lines[i]

        thisLine = thisLine.lstrip().rstrip('\n')
        if thisLine.startswith('//') or thisLine.startswith('/*') or thisLine.startswith('*') or thisLine.startswith('*/'):
            break
        thisLine = thisLine.lstrip()

        try:
            token = list(javalang.tokenizer.tokenize(thisLine))

            thisLine = ''
            for t in token:
                thisLine += t.value
        except:
            pass

        if thisLine.endswith(';') or thisLine.endswith('}') or thisLine.endswith('{'):
            lineStart = i + 1
            break
        else:
            lineStart = i
    lineEnd = line_num
    for i in range(line_num, len(lines)):
        thisLine = lines[i]

        thisLine = thisLine.lstrip().rstrip('\n')
        if thisLine.startswith('//'):
            break
        try:
            token = list(javalang.tokenizer.tokenize(thisLine))
            thisLine = ''
            for t in token:
                thisLine += t.value
        except:
            pass

        if thisLine.endswith(';') or thisLine.endswith('}') or thisLine.endswith('{'):
            lineEnd = i
            break
        else:
            lineEnd = i

    code = ''
    for i in range(lineStart, lineEnd + 1):
        thisLine = lines[i].rstrip('\n').lstrip(' ')
        if i == lineStart:
            if thisLine.startswith('}'):
                thisLine = thisLine[1:]
        code += thisLine

    return code

def mateCode(codeALL, code):
    code = code[:-3]
    try:
        index = codeALL.index(code[:10])
    except:
        return -1
    cntkuohao = 0
    cntzhongkuohao = 0
    cntdakuohao = 0
    cntyinhao = 0
    cntshuangyinhao = 0
    for i in range(len(code)):
        if code[i] == '(':
            cntkuohao += 1
        elif code[i] == ')':
            cntkuohao -= 1
        elif code[i] == '[':
            cntzhongkuohao += 1
        elif code[i] == ']':
            cntzhongkuohao -= 1
        elif code[i] == '{':
            cntdakuohao += 1
        elif code[i] == '}':
            cntdakuohao -= 1
        elif code[i] == '\"':
            cntshuangyinhao += 1
        elif code[i] == '\'':
            cntyinhao += 1
    cntyinhao = cntyinhao % 2
    cntshuangyinhao = cntshuangyinhao % 2
    endInd = index + len(code)
    while 1:
        if cntkuohao < 0 and cntyinhao % 2 == 0 and cntshuangyinhao % 2 == 0:
            endInd -= 1
            break
        if cntzhongkuohao < 0 and cntyinhao % 2 == 0 and cntshuangyinhao % 2 == 0:
            endInd -= 1
            break
        if cntdakuohao < 0 and cntyinhao % 2 == 0 and cntshuangyinhao % 2 == 0:
            endInd -= 1
            break
        if endInd >= len(codeALL):
            endInd -= 1
            break
        if codeALL[endInd] == ')':
            cntkuohao -= 1
        elif codeALL[endInd] == '(':
            cntkuohao += 1
        elif codeALL[endInd] == ']':
            cntzhongkuohao -= 1
        elif codeALL[endInd] == '[':
            cntzhongkuohao += 1
        elif codeALL[endInd] == '}':
            cntdakuohao -= 1
        elif codeALL[endInd] == '{':
            cntdakuohao += 1
        elif codeALL[endInd] == '\"':
            cntshuangyinhao += 1
        elif codeALL[endInd] == '\'':
            cntyinhao += 1
        endInd += 1
    return codeALL[index:endInd]


def slice(Anodes, Aedges, Bnodes, Bedges, nodeIDstart, path, file):
    '''
    "107" -> "109"  [ label = "AST: "]
    "106" -> "112"  [ label = "DDG: &lt;RET&gt;"]
    '''

    tmpEdges = []
    for e in Aedges:
        edgeType = edgeAttr(e)[2]
        if edgeType[:3] in ['AST', 'CDG', 'CFG']:
            tmpEdges.append(e)
        elif edgeType[:3] == 'DDG' and len(edgeType) > 5 and edgeType[5:] != '<RET>' and any(a.isalpha() for a in edgeType[5:]):
            tmpEdges.append(e)
    Aedges = tmpEdges

    tmpEdges = []
    for e in Bedges:
        edgeType = edgeAttr(e)[2]
        if edgeType[:3] in ['AST', 'CDG', 'CFG']:
            tmpEdges.append(e)
        elif edgeType[:3] == 'DDG' and len(edgeType) > 5 and edgeType[5:] != '<RET>' and any(a.isalpha() for a in edgeType[5:]):
            tmpEdges.append(e)
    Bedges = tmpEdges


    nodeID = nodeIDstart + 1
    dictA = {}
    dictB = {}
    CtxNodes = []
    PreNodes = []
    PstNodes = []
    for Bn in Bnodes:
        for An in Anodes:
            AnodeId, AnodeInfo = nodeAttr(An)
            BnodeId, BnodeInfo = nodeAttr(Bn)
            if AnodeInfo == BnodeInfo and len(BnodeInfo) > 0:
                dictA[AnodeId] = nodeID
                dictB[BnodeId] = nodeID
                CtxNodes.append((nodeID, AnodeInfo, 0))
                nodeID += 1
    for An in Anodes:
        AnodeId, AnodeInfo = nodeAttr(An)
        if AnodeId not in dictA and len(AnodeInfo) > 0:
            dictA[AnodeId] = nodeID
            PreNodes.append((nodeID, AnodeInfo, -1))
            nodeID += 1
    for Bn in Bnodes:
        BnodeId, BnodeInfo = nodeAttr(Bn)
        if BnodeId not in dictB and len(BnodeInfo) > 0:
            dictB[BnodeId] = nodeID
            PstNodes.append((nodeID, BnodeInfo, 1))
            nodeID += 1



    newAedges = []
    newBedges = []
    for Ae in Aedges:
        nodeID1, nodeID2, edgeInfo = edgeAttr(Ae)
        if nodeID1 not in dictA.keys() or nodeID2 not in dictA.keys():
            continue
        newAedges.append((dictA[nodeID1], dictA[nodeID2], edgeInfo, 0))
    for Be in Bedges:
        nodeID1, nodeID2, edgeInfo = edgeAttr(Be)
        if nodeID1 not in dictB.keys() or nodeID2 not in dictB.keys():
            continue
        newBedges.append((dictB[nodeID1], dictB[nodeID2], edgeInfo, 0))


    CtxEdges = list(set(newAedges) & set(newBedges))
    PreEdges = []
    PstEdges = []
    newAedges = list(set(newAedges) - set(CtxEdges))
    newBedges = list(set(newBedges) - set(CtxEdges))

    for Ae in newAedges:
        PreEdges.append((Ae[0], Ae[1], Ae[2], -1))
    for Be in newBedges:
        PstEdges.append((Be[0], Be[1], Be[2], 1))


    CtxNodes1 = CtxNodes[:]
    CtxEdges1 = CtxEdges[:]
    backNodes1 = PreNodes[:] + PstNodes[:]
    backEdges1 = PreEdges[:] + PstEdges[:]
    for n2 in backNodes1:
        for e in CtxEdges1 + backEdges1:
            if e[1] == n2[0] and e[2][:3] == 'CDG':
                for n1 in CtxNodes1[:]:
                    if e[0] == n1[0]:
                        CtxNodes1.remove(n1)
                        backNodes1.append(n1)
                        if e in CtxEdges1:
                            CtxEdges1.remove(e)
                        if e not in backEdges1:
                            backEdges1.append(e)

    CtxNodes2 = CtxNodes[:]
    CtxEdges2 = CtxEdges[:]
    backNodes2 = PreNodes[:] + PstNodes[:]
    backEdges2 = PreEdges[:] + PstEdges[:]
    for n2 in backNodes2:
        for e in CtxEdges2 + backEdges2:
            if e[1] == n2[0] and e[2][:3] == 'DDG':
                for n1 in CtxNodes2[:]:
                    if e[0] == n1[0]:
                        CtxNodes2.remove(n1)
                        backNodes2.append(n1)
                        if e in CtxEdges2:
                            CtxEdges2.remove(e)
                        if e not in backEdges2:
                            backEdges2.append(e)

    CtxNodes3 = CtxNodes[:]
    CtxEdges3 = CtxEdges[:]
    backNodes3 = PreNodes[:] + PstNodes[:]
    backEdges3 = PreEdges[:] + PstEdges[:]
    for n2 in backNodes3:
        for e in CtxEdges3 + backEdges3:
            if e[1] == n2[0] and e[2][:3] == 'CFG':
                for n1 in CtxNodes3[:]:
                    if e[0] == n1[0]:
                        CtxNodes3.remove(n1)
                        backNodes3.append(n1)
                        if e in CtxEdges3:
                            CtxEdges3.remove(e)
                        if e not in backEdges3:
                            backEdges3.append(e)

    backNodes = list(set(backNodes1 + backNodes2 + backNodes3))
    backEdges = list(set(backEdges1 + backEdges2 + backEdges3))


    CtxNodes1 = CtxNodes[:]
    CtxEdges1 = CtxEdges[:]
    frontNodes1 = PreNodes[:] + PstNodes[:]
    frontEdges1 = PreEdges[:] + PstEdges[:]
    for n1 in frontNodes1[:]:
        for e in CtxEdges1 + frontEdges1:
            if e[0] == n1[0] and e[2][:3] == 'DDG':
                for n2 in CtxNodes1[:]:
                    if e[1] == n2[0]:
                        CtxNodes1.remove(n2)
                        frontNodes1.append(n2)
                        if e in CtxEdges1:
                            CtxEdges1.remove(e)
                        if e not in frontEdges1:
                            frontEdges1.append(e)


    CtxNodes2 = CtxNodes[:]
    CtxEdges2 = CtxEdges[:]
    frontNodes2 = PreNodes[:] + PstNodes[:]
    frontEdges2 = PreEdges[:] + PstEdges[:]
    for n1 in frontNodes2[:]:
        for e in CtxEdges2 + frontEdges2:
            if e[0] ==n1[0] and e[2][:3] == 'CDG':
                for n2 in CtxNodes2[:]:
                    if e[1] == n2[0]:
                        CtxNodes2.remove(n2)
                        frontNodes2.append(n2)
                        if e in CtxEdges2:
                            CtxEdges2.remove(e)
                        if e not in frontEdges2:
                            frontEdges2.append(e)


    CtxNodes3 = CtxNodes[:]
    CtxEdges3 = CtxEdges[:]
    frontNodes3 = PreNodes[:] + PstNodes[:]
    frontEdges3 = PreEdges[:] + PstEdges[:]
    for n1 in frontNodes3[:]:
        for e in CtxEdges3 + frontEdges3:
            if e[0] ==n1[0] and e[2][:3] == 'CFG':
                for n2 in CtxNodes3[:]:
                    if e[1] == n2[0]:
                        CtxNodes3.remove(n2)
                        frontNodes3.append(n2)
                        if e in CtxEdges3:
                            CtxEdges3.remove(e)
                        if e not in frontEdges3:
                            frontEdges3.append(e)


    frontNodes = list(set(frontNodes1 + frontNodes2 + frontNodes3))
    frontEdges = list(set(frontEdges1 + frontEdges2 + frontEdges3))

    sliceNodes = list(set(backNodes + frontNodes))
    sliceEdges = list(set(backEdges + frontEdges))
    CtxNodes = list(set(CtxNodes) - set(sliceNodes))
    CtxEdges = list(set(CtxEdges) - set(sliceEdges))


    for n2 in sliceNodes:
        for e in CtxEdges + sliceEdges:
            if n2[0] in [e[0], e[1]] and e[2][:3] == 'AST':
                for n1 in CtxNodes:
                    if n1[0] in [e[0], e[1]] and ifSameLine(n1, n2):
                        CtxNodes.remove(n1)
                        sliceNodes.append(n1)
                        if e in CtxEdges:
                            CtxEdges.remove(e)
                        if e not in sliceEdges:
                            sliceEdges.append(e)

    sliceNodes = list(set(sliceNodes))
    sliceEdges = list(set(sliceEdges))


    for e in sliceEdges:
        for n in CtxNodes:
            if n[0] in [e[0], e[1]] and e[2][:3] == 'AST':
                CtxNodes.remove(n)
                sliceNodes.append(n)


    slimNodes = []
    slimEdges = []
    for n in sliceNodes:
        if '</SUB>' not in str(n[1]):
            continue
        i = n[1].index('<SUB>')
        j = n[1].index('</SUB>')
        line_num = n[1][i + 5:j]

        comma1 = n[1].index(',')
        text1 = n[1][1:comma1]
        if text1 == 'BLOCK':
            type = 'block'
            code = 'empty'
        elif text1 == 'LOCAL':
            type = 'local'
            code = n[1][comma1 + 1:i - 1]
        elif text1 == 'MODIFIER':
            type = 'modifier'
            code = n[1][comma1 + 1:i - 1].lower()
        elif text1 == 'CONTROL_STRUCTURE':
            comma2 = n[1].index(',', comma1 + 1)
            text2 = n[1][comma1 + 1:comma2]
            type = 'control structure ' + text2
            code = n[1][comma2 + 1:i - 1]
        elif text1 == 'PARAM':
            type = 'param'
            code = n[1][comma1 + 1:i - 1]
        elif text1 == 'IDENTIFIER':
            comma2 = n[1].index(',', comma1 + 1)
            type = 'identifier'
            code = n[1][comma2 + 1:i - 1]
        elif text1 == 'FIELD_IDENTIFIER':
            type = 'field identifier'
            text2 = n[1][comma1 + 1:i - 1]
            code = text2[:int(len(text2) / 2)]
        elif text1 == 'LITERAL':
            comma2 = n[1].index(',', comma1 + 1)
            type = 'literal'
            code = n[1][comma2 + 1:i - 1]
        elif text1 == 'METHOD':
            type = 'method'
            code = n[1][comma1 + 1:i - 1]
        elif text1 == 'METHOD_RETURN':
            type = 'method return'
            code = n[1][comma1 + 1:i - 1]
        elif text1 == 'RETURN':
            type = 'return'
            text2 = n[1][comma1 + 1:i - 1]
            code = text2[:int(len(text2) / 2)]
        elif text1 == 'ANNOTATION':
            comma2 = n[1].index(',', comma1 + 1)
            type = 'annotation'
            code = n[1][comma2 + 1:i - 1]
        elif text1 == 'OR':
            type = 'or'
            code = n[1][comma1 + 1:i - 1]
        elif text1 == 'TYPE_REF':
            comma2 = n[1].index(',', comma1 + 1)
            type = 'type refertence'
            code = n[1][comma2 + 1:i - 1]
        elif text1 == 'UNKNOWN':
            comma2 = n[1].index(',', comma1 + 1)
            type = 'unknown'
            code = n[1][comma2 + 1:i - 1]
        elif text1.startswith('<operator>'):
            type = 'operator ' + text1[11:]
            code = n[1][comma1 + 1:i - 1]
        else:
            type = 'empty'
            code = n[1][comma1 + 1:i - 1]

        if code.startswith('//') or '<empty>' in code or code == '' or code == '\n' or code == '\r\n':
            continue

        if code.endswith('...'):
            if n[-1] == -1:
                filepath = path + '/a/' + file + '.java'
                codeALL = codeCompletion(filepath, int(line_num) - 1)
            else:
                filepath = path + '/b/' + file + '.java'
                codeALL = codeCompletion(filepath, int(line_num) - 1)
            code = mateCode(codeALL, code)

        if code == -1:
            continue

        if '\\012' in code:
            code = code.replace("\\012", '')
        try:
            token = list(javalang.tokenizer.tokenize(code + ';'))
            if len(token) == 1 or len(token) == 0:
                continue
        except:
            print(file + '中' + code + 'cannot parse by javalang')

        slimNodes.append((n[0], n[-1], line_num, type, code))


    for e in sliceEdges:
        flag = 0
        for n in slimNodes:
            if n[0] == e[0]:
                flag += 1
            if n[0] == e[1]:
                flag += 1
        if flag == 2:
            edgeType = e[2][:3]
            if e[0] == e[1]:
                continue
            slimEdges.append((e[0], e[1], edgeType, e[3]))
        else:
            continue

    slimNodes = list(set(slimNodes))
    slimEdges = list(set(slimEdges))



    for n in slimNodes[:]:
        if n[1] != 0:
            continue
        for n1 in slimNodes[:]:
            if n1[1] == -1 and n1[2] == n[2] and n1[3] == n[3] and n1[4] == n[4]:
                for n2 in slimNodes[:]:
                    if n2[1] == 1 and n2[2] == n[2] and n2[3] == n[3] and n2[4] == n[4]:
                        if n not in slimNodes:
                            continue
                        slimNodes.remove(n)
                        for e in slimEdges[:]:
                            if e[0] == n[0]:
                                if e[-1] == -1:
                                    slimEdges.remove(e)
                                    slimEdges.append((n1[0], e[1], e[2], e[3]))
                                elif e[-1] == 1:
                                    slimEdges.remove(e)
                                    slimEdges.append((n2[0], e[1], e[2], e[3]))
                            if e[1] == n[0]:
                                if e[-1] == -1:
                                    slimEdges.remove(e)
                                    slimEdges.append((e[0], n1[0], e[2], e[3]))
                                elif e[-1] == 1:
                                    slimEdges.remove(e)
                                    slimEdges.append((e[0], n2[0], e[2], e[3]))
    slimEdges = list(set(slimEdges))


    usedNodes = []
    for e in slimEdges[:]:
        usedNodes.append(e[0])
        usedNodes.append(e[1])
    for n in slimNodes[:]:
        if n[0] not in usedNodes and n[1] == 0:
            slimNodes.remove(n)


    nodeDegree = {}
    for e in slimEdges[:]:
        if e[0] not in nodeDegree.keys():
            nodeDegree[e[0]] = 1
        else:
            nodeDegree[e[0]] += 1
        if e[1] not in nodeDegree.keys():
            nodeDegree[e[1]] = 1
        else:
            nodeDegree[e[1]] += 1


    line_maxLen = {}
    for n in slimNodes[:]:
        if n[2] not in nodeDegree.keys():
            line_maxLen[n[2]] = len(n[4])
            continue
        t = line_maxLen[n[2]]
        line_maxLen[n[2]] = max(len(n[4]), t)
    for n in slimNodes[:]:
        if n[0] not in nodeDegree.keys() and line_maxLen[n[2]] > len(n[4]):
            slimNodes.remove(n)
    slimNodes = list(set(slimNodes))


    reserveNodes = []
    for n in slimNodes:
        if n[0] in nodeDegree.keys():
            reserveNodes.append((n[0], n[1], nodeDegree[n[0]], n[2], n[3], n[4]))
        else:
            reserveNodes.append((n[0], n[1], 0, n[2], n[3], n[4]))


    nodeIDList = []
    for n in slimNodes[:]:
        nodeIDList.append(n[0])
    for e in slimEdges[:]:
        if e[0] not in nodeIDList or e[1] not in nodeIDList:
            slimEdges.remove(e)


    reserveEdges = []
    dictEdges = {}
    dictType = {'AST': 'A', 'CDG': 'C', 'CFG': 'F', 'DDG': 'D'}
    for e1 in slimEdges:
        if (e1[0], e1[1]) in dictEdges.keys():
            continue
        edgeType = e1[2][0]
        for e2 in slimEdges:
            if e1[0] == e2[0] and e1[1] == e2[1] and e1[2] != e2[2]:
                edgeType += dictType[e2[2]]

        edgeType = ''.join(sorted(edgeType))
        dictEdges[(e1[0], e1[1])] = edgeType
        reserveEdges.append((e1[0], e1[1], edgeType, e1[3]))

    return reserveNodes, reserveEdges, nodeID



def mergeCPG(path, out_path, label):
    AnodeByFunc, AedgeByFunc, AFuncFile = importCPG(path + '/outA/')
    BnodeByFunc, BedgeByFunc, BFuncFile = importCPG(path + '/outB/')
    ABFunc = [f for f in AnodeByFunc.keys() if f in BnodeByFunc.keys()]
    AFunc = [f for f in AnodeByFunc.keys() if f not in BnodeByFunc.keys()]
    BFunc = [f for f in BnodeByFunc.keys() if f not in AnodeByFunc.keys()]

    nodeIDstart = 1
    allNodes = []
    allEdges = []

    for func in ABFunc:
        Anodes = AnodeByFunc[func]
        Aedges = AedgeByFunc[func]
        Bnodes = BnodeByFunc[func]
        Bedges = BedgeByFunc[func]
        file = BFuncFile[func]
        sliceNodes, sliceEdges, nodeID = slice(Anodes, Aedges, Bnodes, Bedges, nodeIDstart, path, file)
        if len(sliceNodes) > 0 and len(sliceEdges) > 0:
            allNodes += sliceNodes
            allEdges += sliceEdges
            nodeIDstart = nodeID
    for func in AFunc:
        Anodes = AnodeByFunc[func]
        Aedges = AedgeByFunc[func]
        Bnodes = []
        Bedges = []
        file = AFuncFile[func]
        sliceNodes, sliceEdges, nodeID = slice(Anodes, Aedges, Bnodes, Bedges, nodeIDstart, path, file)
        if len(sliceNodes) > 0 and len(sliceEdges) > 0:
            allNodes += sliceNodes
            allEdges += sliceEdges
            nodeIDstart = nodeID
    for func in BFunc:
        Anodes = []
        Aedges = []
        Bnodes = BnodeByFunc[func]
        Bedges = BedgeByFunc[func]
        file = BFuncFile[func]
        sliceNodes, sliceEdges, nodeID = slice(Anodes, Aedges, Bnodes, Bedges, nodeIDstart, path, file)
        if len(sliceNodes) > 0 and len(sliceEdges) > 0:
            allNodes += sliceNodes
            allEdges += sliceEdges
            nodeIDstart = nodeID

    if len(allNodes) > 0 or len(allEdges) > 0:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        f = open(out_path + '/out.log', 'w+')
        f.write('label ' + label + '\n')
        f.write('nodes' + '\n')
        f.write('\n'.join(map(str, allNodes)))
        f.write('\n' + 'edges' + '\n')
        f.write('\n'.join(map(str, allEdges)))
        f.close()
        return 1
    return 0


def main():
    logsPath = './logs'
    logfile = 'cpgMerge.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))


    root = 'dataset/positive'
    label = '1'
    for commit in os.listdir(root):
        path = root + '/' + commit
        out_path = './data/graph/' + commit
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            flag = mergeCPG(path, out_path, label)
            if flag == 0:
                os.removedirs(out_path)
                print(commit + 'slicing is empty')
            else:
                print(commit + 'CPG slice successfully')
        else:
            print(commit + 'already slice')

    root = 'dataset/negative'
    label = '0'
    for commit in os.listdir(root):
        path = root + '/' + commit
        out_path = './data/graph/' + commit
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            flag = mergeCPG(path, out_path, label)
            if flag == 0:
                os.removedirs(out_path)
                print(commit + 'slicing is empty')
            else:
                print(commit + 'CPG slice successfully')
        else:
            print(commit + 'already slice')


if __name__ == '__main__':
    main()

