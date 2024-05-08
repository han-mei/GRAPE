@main def exec(outFile: String) = {
   importCpg("cpg.bin")
   cpg.method.map(x=>(x.name, x.lineNumber, x.lineNumberEnd)).toJson #> outFile
}