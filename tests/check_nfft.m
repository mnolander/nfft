(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



AppendTo[$Path, NotebookDirectory[]];
<<PrintVector`


P=64;(* Working precision. *)


GenerateFilename[prefix_][NN_,M_]:=Module[{d=Length[NN], (* Dimension. *)},Return[FileNameJoin[{prefix,"nfft_"<>ToString[d]<>"d_"<>StringJoin[Map[Function[x,ToString[x]<>"_"],NN]<>ToString[M]<>".txt"]}]]]
GenerateFilenameAdjoint[prefix_][NN_,M_]:=Module[{d=Length[NN], (* Dimension. *)},Return[FileNameJoin[{prefix,"nfft_adjoint_"<>ToString[d]<>"d_"<>StringJoin[Map[Function[x,ToString[x]<>"_"],NN]<>ToString[M]<>".txt"]}]]]
Generate[NN_,M_,FilenameGenerator_]:=Module[
{
d=Length[NN], (* Dimension. *)
file
},
SeedRandom[1];
II=Table[Table[k,{k,Ceiling[-NN[[i]]/2],Floor[(NN[[i]]-1)/2]}],{i,1,d}];
II[[0]]=Sequence;
II=Flatten[Outer[List,II],d-1];(* Index set. *)
x =Transpose[ Table[RandomReal[{-1/2,1/2},M,WorkingPrecision->P],{i,1,d}]] ;(* Random nodes. *)
fhat = (*Table[If[i==1,1,0],{i,1,Length[II]}]*) RandomComplex[{-1-I,1+I},Length[II],WorkingPrecision->P]; (* Random Fourier coefficients. *)
f=Table[Sum[fhat[[k]]*Exp[-2*\[Pi]*I*Dot[II[[k]],x[[j]]]],{k,1,Length[II]}],{j,1,M}];(* Function values. *)
filename=FilenameGenerator[NN,M];
file = OpenWrite[filename];
WriteString[file, FormatIntegerRaw[d] <>"\n"];
WriteString[file, FormatIntegerVectorRaw[NN]<>"\n"];
WriteString[file, FormatIntegerRaw[M]<>"\n"];
WriteString[file, FormatVectorRaw[Flatten[x]]<>"\n"];
WriteString[file, FormatVectorRaw[fhat]<>"\n"];
WriteString[file, FormatVectorRaw[f]<>"\n"];
Close[file];
(*Print[filename];*)
Return[{FileBaseName[FileNameTake[filename,-1]], FileNameTake[filename,-2]}]
]
GenerateAdjoint[NN_,M_,FilenameGenerator_]:=Module[
{
d=Length[NN], (* Dimension. *)
file
},
SeedRandom[1];
II=Table[Table[k,{k,Ceiling[-NN[[i]]/2],Floor[(NN[[i]]-1)/2]}],{i,1,d}];
II[[0]]=Sequence;
II=Flatten[Outer[List,II],d-1];(* Index set. *)
x=Transpose[Table[RandomReal[{-1/2,1/2},M,WorkingPrecision->P],{i,1,d}]] ;(* Random nodes. *)
f=RandomComplex[{-1-I,1+I},M,WorkingPrecision->P];(* Random function values. *)
fhat=Table[Sum[f[[j]]*Exp[2*\[Pi]*I*Dot[II[[k]],x[[j]]]],{j,1,M}],{k,1,Length[II]}];(* Pseudo Fourier coefficients. *)
filename=FilenameGenerator[NN,M];
file = OpenWrite[filename];
WriteString[file, FormatIntegerRaw[d] <>"\n"];
WriteString[file, FormatIntegerVectorRaw[NN]<>"\n"];
WriteString[file, FormatIntegerRaw[M]<>"\n"];
WriteString[file, FormatVectorRaw[Flatten[x]]<>"\n"];
WriteString[file, FormatVectorRaw[fhat]<>"\n"];
WriteString[file, FormatVectorRaw[f]<>"\n"];
Close[file];
(*Print[filename];*)
Return[{FileBaseName[FileNameTake[filename,-1]], FileNameTake[filename,-2]}]
]
MakeTestcase[NN_,M_]:=Generate[NN,M,GenerateFilename[FileNameJoin[{NotebookDirectory[],"data"}]]]
MakeTestcaseAdjoint[NN_,M_]:=GenerateAdjoint[NN,M,GenerateFilenameAdjoint[FileNameJoin[{NotebookDirectory[],"data"}]]]
Formatter[x_]:="&"<>x;



