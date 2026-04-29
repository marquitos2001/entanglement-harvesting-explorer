(* ::Package:: *)

(* ::Section:: *)
(*Evaluate and export*)


(* ::Text:: *)
(*This code is the second of three programs that provide entanglement harvesting plots for arbitrary switching functions . It is based on and follows the method from M . Morote - Balboa, T . Rick Perche, "Optimization of entanglement harvesting with arbitrary temporal profiles: the limit of second order perturbation theory", arXiv : 2604.06303 (2026) . *)
(**)
(*This code takes the monomial - basis matrices Subscript[H, nm] , Subscript[W, nm] and Subscript[\[CapitalDelta], nm] (Eq. (41)), produced by the first program (as .mx files), substitutes numerical values for T and L (keeping  \[CapitalOmega] symbolic), transforms them to the Hermite basis via \!\(\*SubsuperscriptBox[\(T\), \(t\), \(T\)]\). M(\[CapitalOmega]). Subscript[T, t] \:200b, evaluates at a grid of \[CapitalOmega] values, and exports the results as HDF5 (.h5) files for downstream analysis (e . g ., in Python). *)
(**)
(*It is structured as follows :*)
(**)
(*     1. Setup : The user sets the directory (targetDir) where files are saved and read from.*)
(**)
(*     2. Functions : Different functions are introduced : *)
(*     *)
(*        -  Helper functions for the Hermite basis : getPrec (adaptive precision, giving the number of digits of precision at a given N to avoid numerical noise), getT (optimal Hermite width ensuring compact support) (Eqs.(46)-(47)), and generateTt (builds the transformation matrix to the Hermite basis) . *)
(*        *)
(*        - Core function: parallelSubExpand takes a symbolic matrix depending on { T, \[CapitalOmega] ,  L}, substitutes numerical via disk - based parallel chunking (to avoid memory bottlenecks from distributing large symbolic expressions to all kernels), and returns a matrix depending only on \[CapitalOmega].*)
(*        *)
(*     3. Configuration: The user sets the physical parameters.*)
(**)
(*     4. Main evaluation loop : For each truncation order n in nSteps:*)
(**)
(*       - Compute the Hermite width parameter Subscript[T, N] and the transformation matrix \!\(\*SubsuperscriptBox[\(T\), \(t\), \(T\)]\)*)
(**)
(*       - Use parallelSubExpand to substitute T\[RightArrow]Subscript[T, N], L\[RightArrow] Lval into the matrices*)
(**)
(*       - Build functions M(\[CapitalOmega]) from the resulting expressions and distribute them to all parallel kernels via disk . Evaluate at all \[CapitalOmega] values in parallel, applying the Hermite basis transformation \!\(\*SubsuperscriptBox[\(T\), \(t\), \(T\)]\). M(\[CapitalOmega]). Subscript[T, t] at each point.*)
(* *)
(*       - Export the Hermite - basis matrices (real and imaginary parts) to an HDF5 file . *)
(*       *)
(*      *)
(*  Important notes: *)
(*  *)
(*     - This code requires the .mx files produced by the first program (compute_derivatives.wl) to be present in targetDir . These files must have been produced with the same version of Mathematica . *)


(* ::Subsection::Closed:: *)
(*1. Setup*)


(*Assumptions*)
$Assumptions = {T > 0, \[Sigma] > 0, \[CapitalOmega] > 0,a \[Element] Reals, b \[Element] Reals, L > 0};


(*Path to the directory where the files will be saved*)
targetDir = "replace/with/path/"; 
SetDirectory[targetDir];


(* ::Subsection::Closed:: *)
(*2. Functions*)


(*Adaptive precision to avoid numerical noise*)
getPrec[n_] := Max[20, Round[n * 1.7 + 20]];


(*Scaling function for the optimal Hermite width parameter*)
f[n_] := Sqrt[2 n + 1] / (Sqrt[2 n + 1] + 2);

(*Optimal Hermite width parameter T_n for truncation order n*)
getT[n_, currentPrec_] := SetPrecision[(5/2 f[n]) / Sqrt[2 n + 1], currentPrec];



(*Transformation matrix to Hermite basis*)
generateTt[n_, TvalIn_] := Module[
  {prec, coeffs, norm, mat, c, invT, invTpow},
  prec = Precision[TvalIn];

  (*Step 1: Build Hermite polynomial coefficients via recurrence
    H_0(x) = 1,  H_1(x) = 2x,  H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x) *)
  coeffs = Table[{}, {n + 1}];
  coeffs[[1]] = {1};
  If[n >= 1, coeffs[[2]] = {0, 2}];
  Do[
    c = Table[0, {idx + 1}];
    Do[c[[k + 1]] += 2 coeffs[[idx]][[k]], {k, 1, idx}];        (*2x \[CenterDot] H_{n-1}*)
    Do[c[[k]] -= 2 (idx - 1) coeffs[[idx - 1]][[k]], {k, 1, idx - 1}]; (*-2(n-1) \[CenterDot] H_{n-2}*)
    coeffs[[idx + 1]] = c;
  , {idx, 2, n}];

  (*Step 2: Build powers of 1/T for the rescaling H_n(x/T)*)
  invT = SetPrecision[1 / TvalIn, prec];
  invTpow = Table[0, {n + 1}];
  invTpow[[1]] = SetPrecision[1, prec];
  Do[invTpow[[k + 1]] = invTpow[[k]] * invT, {k, 1, n}];

  (*Step 3: Build normalization factors 1/Sqrt[2^n n! Sqrt[\[Pi]] T] incrementally*)
  norm = Table[0, {n + 1}];
  norm[[1]] = SetPrecision[1 / Sqrt[Sqrt[SetPrecision[\[Pi], prec]] * TvalIn], prec];
  Do[norm[[idx + 1]] = norm[[idx]] / Sqrt[SetPrecision[2 idx, prec]], {idx, 1, n}];

  (*Step 4: Assemble the matrix T_t*)
  mat = Table[SetPrecision[0, prec], {n + 1}, {n + 1}];
  Do[
    c = coeffs[[idx + 1]];
    Do[
      If[c[[k + 1]] != 0,
        mat[[k + 1, idx + 1]] =
          SetPrecision[c[[k + 1]], prec] * norm[[idx + 1]] * invTpow[[k + 1]];
      ];
    , {k, 0, idx}];
  , {idx, 0, n}];
  mat
];


(* --- 5b. parallelSubExpand: substitute T and L in parallel via disk-based chunking ---

     This function takes a symbolic (Ntarget+1)\[Times](Ntarget+1) matrix, extracts the
     top-left (n+1)\[Times](n+1) block, and substitutes T \[RightArrow] Tn, L \[RightArrow] Lval. The result
     is a matrix depending only on \[CapitalOmega].

     Arguments:
       sym        : the full symbolic matrix (e.g., fullGsym)
       label      : display name for progress messages
       chunkType  : file tag for input chunks
       resultType : file tag for output chunks                                     *)

  parallelSubExpand[sym_, label_, chunkType_, resultType_] := Module[
    {block, rowsPerChunk, nChunks, result, r1i, r2i},

    (*Extract the (n+1)\[Times](n+1) top-left block \[LongDash] no need to process the full 201\[Times]201*)
    block        = sym[[1 ;; nRows, 1 ;; nRows]];

    (*Split into approximately one chunk per kernel*)
    rowsPerChunk = Max[1, Ceiling[nRows / $KernelCount]];
    nChunks      = Ceiling[nRows / rowsPerChunk];
    Print[" ", label, ": ", nChunks, " chunks (", rowsPerChunk, " rows each)"];

    (*Write chunks to disk*)
    t1 = AbsoluteTime[];
    Do[
      r1i = (ci - 1) * rowsPerChunk + 1;
      r2i = Min[ci * rowsPerChunk, nRows];
      Export[chunkFile[runTag, n, ci, chunkType], {r1i, r2i, block[[r1i ;; r2i]], Tn}];
    , {ci, 1, nChunks}];
    Clear[block];
    Print["  chunks written: ", Round[AbsoluteTime[] - t1, 0.1], "s"];

    (*Each kernel reads its chunk, substitutes T \[RightArrow] Tn and L \[RightArrow] Lval, expands, and writes back.
      After substitution, the only remaining symbolic variable is \[CapitalOmega].*)
    t1 = AbsoluteTime[];
    ParallelDo[
      Module[{data, pr1, pr2, chunk, tn, res, ifile, ofile},
        ifile = chunkFile[runTag, n, ci, chunkType];
        If[! FileExistsQ[ifile],
          Print["  ERROR: missing chunk ", ci, " (", label, ") n=", n];
          Return[Null, Module];
        ];
        data = Import[ifile];
        {pr1, pr2, chunk, tn} = data;
        res   = Expand[chunk /. {T -> tn, L -> Lval}];   (*Substitute T, L; keep \[CapitalOmega] symbolic*)
        ofile = chunkFile[runTag, n, ci, resultType];
        Export[ofile, {pr1, pr2, res}];
        Quiet[DeleteFile[ifile]];                         (*Clean up input chunk*)
        Clear[data, chunk, res, tn];
      ],
      {ci, 1, nChunks},
      Method -> "CoarsestGrained"
    ];
    Print["  parallel sub+expand: ", Round[AbsoluteTime[] - t1, 0.1], "s"];

    (*Reassemble the full matrix from result chunks*)
    t1     = AbsoluteTime[];
    result = Table[SetPrecision[0, currentPrec], {nRows}, {nRows}];
    Do[
      Module[{rfile, pr1, pr2, res},
        rfile = chunkFile[runTag, n, ci, resultType];
        If[! FileExistsQ[rfile],
          Print["  ERROR: missing result ", ci, " (", label, ") n=", n];
          ,
          {pr1, pr2, res} = Import[rfile];
          result[[pr1 ;; pr2]] = res;
          Quiet[DeleteFile[rfile]];    (*Clean up result chunk*)
        ];
      ];
    , {ci, 1, nChunks}];
    Print["  results collected: ", Round[AbsoluteTime[] - t1, 0.1], "s"];

    result    (*Returns an (n+1)\[Times](n+1) matrix depending only on \[CapitalOmega]*)
  ];


(* ::Subsection::Closed:: *)
(*3. Configuration*)


(*Truncation N of the Hermite basis \[LongDash] must match the .mx files from the first program*)
Ntarget = 200;

(*Physical parameters*)
Lval        = 5;                                     (*Detector separation*)
Tval        = getT[Ntarget, getPrec[Ntarget]];      (*Hermite width parameter for spacelike separation. If the user wants to allow some signalling(Sec.V B), set value by hand (eg. 1)*)

(*Grid of energy gap values \[CapitalOmega] to evaluate*)
omegaValues = Range[0, 20, 1/100];  (*IMPORTANT: the step must be written as a fraction to keep the precision*)

(*Truncation orders to export *)
nSteps      = {150};


(* ::Subsection::Closed:: *)
(*4. Main evaluation*)


(* ::Subsubsection::Closed:: *)
(*Parallel evaluation*)


(*Utility: generate file paths for disk-based parallel chunks*)
chunkFile[tag_, nv_, ci_, type_] :=
  FileNameJoin[{targetDir,
    tag <> "_" <> type <> "_n" <> ToString[nv] <> "_c" <> ToString[ci] <> ".mx"}];

(*Maximum number of parallel kernels to launch*)
maxKernels = $ProcessorCount;

launchAllKernels[] := (
  Quiet[CloseKernels[]];
  LaunchKernels[maxKernels];
  Print["Launched ", $KernelCount, " parallel kernels (requested ", maxKernels, ")."];
);

(*Load the three monomial-basis derivative matrices produced by the first program.
  These are (Ntarget+1) \[Times] (Ntarget+1) symbolic matrices depending on {T, \[CapitalOmega], L}.
  They are loaded once and kept in memory for all truncation orders.
  Note: fullGsym holds the H matrix, fullWsym the W matrix, fullDsym the \[CapitalDelta] matrix.*)
Print["Loading derivative matrices (one-time)..."];
tLoad0 = AbsoluteTime[];
fullGsym = Import["DerivativesHTimeAux"     <> ToString[Ntarget] <> ".mx"];
fullWsym = Import["DerivativesWmpTimeAux"   <> ToString[Ntarget] <> ".mx"];
fullDsym = Import["DerivativesDeltaTimeAux" <> ToString[Ntarget] <> ".mx"];
Print["Loaded H, W, Delta in ", Round[AbsoluteTime[] - tLoad0, 0.1], " s"];

launchAllKernels[];
nKernels = Length[Kernels[]];
ParallelEvaluate[SetDirectory[targetDir]];

nOmega    = Length[omegaValues];
stepTimes = {};
Print["Starting Export Loop..."];
globalStart = AbsoluteTime[];


(* ::Subsubsection::Closed:: *)
(*Main loop*)


Do[
  Print["--------------------------------------"];
  Print["Starting n=", n];
  stepStart   = AbsoluteTime[];
  currentPrec = getPrec[n];
  Print[" Precision: ", currentPrec];

  (* --- 5a. Compute the Hermite width and basis-change matrix --- *)
  Tn = SetPrecision[Tval, currentPrec];
  Print[" T = ", N[Tn]];

  t1  = AbsoluteTime[];
  Ttn = generateTt[n, Tn];    (*Build T_t: (n+1)\[Times](n+1) change-of-basis matrix*)
  Print[" generateTt done: ", Round[AbsoluteTime[] - t1, 0.1], "s"];

  t1   = AbsoluteTime[];
  TtnT = Transpose[Ttn];      (*T_t^\[DownTee] for the basis transformation T_t^\[DownTee] \[CenterDot] M \[CenterDot] T_t*)
  Print[" Transpose done: ", Round[AbsoluteTime[] - t1, 0.1], "s"];

  nRows = n + 1;

  (* --- 5c. Substitute T and L into all three matrices --- *)

  tSubG      = AbsoluteTime[];
  GOmegaOnly = parallelSubExpand[fullGsym, "H", "hchunk", "hresult"];
  Print[" H total: ", Round[AbsoluteTime[] - tSubG, 0.1], "s"];

  tSubW      = AbsoluteTime[];
  WOmegaOnly = parallelSubExpand[fullWsym, "W", "wchunk", "wresult"];
  Print[" W total: ", Round[AbsoluteTime[] - tSubW, 0.1], "s"];

  tSubD      = AbsoluteTime[];
  DOmegaOnly = parallelSubExpand[fullDsym, "D", "dchunk", "dresult"];
  Print[" Delta total: ", Round[AbsoluteTime[] - tSubD, 0.1], "s"];

  (* --- 5d. Build pure functions of \[CapitalOmega] and distribute to kernels ---

     The \[CapitalOmega] variable is renamed to \[CapitalOmega]v to avoid conflicts with global symbols.
     Each matrix expression and the basis-change data are saved to disk,
     then each kernel loads them and compiles a Function[\[CapitalOmega]v, matrix-expression].
     When called with a numerical \[CapitalOmega], this function just plugs in the number
     with no further symbolic processing.                                        *)

  evalStart  = AbsoluteTime[];
  gExprReady = GOmegaOnly /. \[CapitalOmega] -> \[CapitalOmega]v;
  wExprReady = WOmegaOnly /. \[CapitalOmega] -> \[CapitalOmega]v;
  dExprReady = DOmegaOnly /. \[CapitalOmega] -> \[CapitalOmega]v;
  Clear[GOmegaOnly, WOmegaOnly, DOmegaOnly];
  omegaHP = SetPrecision[omegaValues, currentPrec];

  (*Save expressions to disk for kernel distribution*)
  tempGexpr = FileNameJoin[{targetDir, runTag <> "_Gexpr_n" <> ToString[n] <> ".mx"}];
  tempWexpr = FileNameJoin[{targetDir, runTag <> "_Wexpr_n" <> ToString[n] <> ".mx"}];
  tempDexpr = FileNameJoin[{targetDir, runTag <> "_Dexpr_n" <> ToString[n] <> ".mx"}];
  tempMisc  = FileNameJoin[{targetDir, runTag <> "_misc_n"  <> ToString[n] <> ".mx"}];

  Export[tempGexpr, gExprReady];
  Export[tempWexpr, wExprReady];
  Export[tempDexpr, dExprReady];
  Export[tempMisc,  {Ttn, TtnT, omegaHP}];
  Clear[gExprReady, wExprReady, dExprReady, Ttn, TtnT, omegaHP];

  (*Each kernel loads the expressions from disk and compiles them into pure functions*)
  tLoad = AbsoluteTime[];
  ParallelEvaluate[
    localGexpr = Import[FileNameJoin[{Directory[], runTag <> "_Gexpr_n" <> ToString[n] <> ".mx"}]];
    localWexpr = Import[FileNameJoin[{Directory[], runTag <> "_Wexpr_n" <> ToString[n] <> ".mx"}]];
    localDexpr = Import[FileNameJoin[{Directory[], runTag <> "_Dexpr_n" <> ToString[n] <> ".mx"}]];
    {localTtn, localTtnT, localOmHP} =
      Import[FileNameJoin[{Directory[], runTag <> "_misc_n" <> ToString[n] <> ".mx"}]];
    localGfn = Function[\[CapitalOmega]v, Evaluate[localGexpr]];
    localWfn = Function[\[CapitalOmega]v, Evaluate[localWexpr]];
    localDfn = Function[\[CapitalOmega]v, Evaluate[localDexpr]];
    Clear[localGexpr, localWexpr, localDexpr];
  ];
  loadTime = AbsoluteTime[] - tLoad;

  (* --- 5e. Evaluate at all \[CapitalOmega] values with Hermite basis transformation ---

     For each \[CapitalOmega]_k, compute T_t^\[DownTee] \[CenterDot] M(\[CapitalOmega]_k) \[CenterDot] T_t (Eq. (40)).
     This transforms from the monomial basis to the Hermite basis.
     The 2001 evaluations are embarrassingly parallel \[LongDash] one per \[CapitalOmega] value.  *)

  stackG = ParallelTable[
    localTtnT . localGfn[localOmHP[[k]]] . localTtn,
    {k, 1, nOmega}, Method -> "CoarsestGrained"];
  stackW = ParallelTable[
    localTtnT . localWfn[localOmHP[[k]]] . localTtn,
    {k, 1, nOmega}, Method -> "CoarsestGrained"];
  stackD = ParallelTable[
    localTtnT . localDfn[localOmHP[[k]]] . localTtn,
    {k, 1, nOmega}, Method -> "CoarsestGrained"];

  (*Clean up kernel-local variables and temporary files*)
  ParallelEvaluate[Clear[localGfn, localWfn, localDfn, localTtn, localTtnT, localOmHP]];
  Quiet[DeleteFile[{tempGexpr, tempWexpr, tempDexpr, tempMisc}]];
  evalTime = AbsoluteTime[] - evalStart;

  Print[" n=", n, " MinPrec G: ", Min[Precision /@ Flatten[stackG]],
    "  MinPrec D: ", Min[Precision /@ Flatten[stackD]],
    " load=", Round[loadTime, 0.1], "s"];

  (* --- 5f. Export to HDF5 ---

     Real and imaginary parts are stored separately because HDF5 does not
     natively support complex numbers. In Python: H = Hreal + 1j*Himag.
     Extended precision is converted to machine precision (float64) for export. *)

  filename = runTag <> "_data_n_" <> ToString[n] <> ".h5";
  If[FileExistsQ[filename], DeleteFile[filename]];
  stackGmach = N[stackG];
  stackWmach = N[stackW];
  stackDmach = N[stackD];
  Clear[stackG, stackW, stackD];

  Export[filename, N[omegaValues],  {"Datasets", "/omegas"}];
  Export[filename, Re[stackGmach],  {"Datasets", "/Hreal"},  "OverwriteTarget" -> "Append"];
  Export[filename, Im[stackGmach],  {"Datasets", "/Himag"},  "OverwriteTarget" -> "Append"];
  Export[filename, Re[stackWmach],  {"Datasets", "/Wreal"},  "OverwriteTarget" -> "Append"];
  Export[filename, Im[stackWmach],  {"Datasets", "/Wimag"},  "OverwriteTarget" -> "Append"];
  Export[filename, Re[stackDmach],  {"Datasets", "/Dreal"},  "OverwriteTarget" -> "Append"];
  Export[filename, Im[stackDmach],  {"Datasets", "/Dimag"},  "OverwriteTarget" -> "Append"];
  Export[filename, {N[Tn]},         {"Datasets", "/T"},       "OverwriteTarget" -> "Append"];
  Export[filename, {n},             {"Datasets", "/n"},       "OverwriteTarget" -> "Append"];
  Export[filename, {Lval},          {"Datasets", "/L"},       "OverwriteTarget" -> "Append"];
  Clear[stackGmach, stackWmach, stackDmach];

  stepTime = AbsoluteTime[] - stepStart;
  AppendTo[stepTimes, {n, stepTime}];
  Print[" Done n=", n, " load=", Round[loadTime, 0.1],
    "s eval=", Round[evalTime, 0.1], "s total=", Round[stepTime, 0.1], "s"];

, {n, nSteps}];


(* ::Subsection::Closed:: *)
(*5. Clean-up*)


Clear[fullGsym, fullWsym, fullDsym];
totalTime = AbsoluteTime[] - globalStart;
CloseKernels[];
Print["======================================"];
Print["ALL EXPORTS COMPLETE - runTag: ", runTag];
Print["T = ", Tval, "  L = ", Lval];
Print["Total Runtime: ", Round[totalTime, 1], " s (",
  Round[totalTime / 60, 0.1], " min)"];
