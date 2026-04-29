(* ::Package:: *)

(* ::Section:: *)
(*Compute derivatives*)


(* ::Text:: *)
(*This code is the first of three programs that provide entanglement harvesting plots for arbitrary switching functions. It is based on and follows the method from M. Morote-Balboa, T. Rick Perche, "Optimization of entanglement harvesting with arbitrary temporal profiles: the limit of second order perturbation theory",  arXiv:2604.06303 (2026). *)
(**)
(*This code exports the (N+1)x(N+1) matrices Subscript[H, nm] , Subscript[W, nm] and Subscript[\[CapitalDelta], nm] (Eq. (41)) as three .mx files, one per matrix. It starts by computing lists of derivatives of the functions that the propagators factorize into (Eqs. (B12)-(B14)), then distributes these lists to the kernels and builds upper-triangle matrices to avoid unnecessary computations by making use of the symmetry/hermicity of the propagators, and finally creates the final matrices Subscript[H, nm] , Subscript[W, nm] and Subscript[\[CapitalDelta], nm].*)
(*It is structured as follows:*)
(**)
(*     1. Setup: The user sets the truncation N (Ntarget) of the Hermite basis Subscript[{\!\(TraditionalForm\`*)
(*\*SubscriptBox[\(\[Chi]\), \(n\)]\)}, n<=N] and the directory (targetDir) where the files will be saved.*)
(*     *)
(*     2. Functions: Different functions are introduced:*)
(*     *)
(*        - Generating functions H(\[Alpha], \[Beta]), W(\[Alpha], \[Beta]) and \[CapitalDelta](\[Alpha], \[Beta]) (Eqs. (A7)-(A9))*)
(*               *)
(*        - The main function (runBatchedLoop) which is used to compute the derivatives of the generators (Eqs. (B6)-(B8)). It takes two lists of derivatives (dFList, dGList) corresponding to the different derivatives of the two simple  functions (F, G) that the generators factorized into (Eqs. (B12)-(B14)). Then, it distributes these lists by batches to the different kernels (in order to save checkpoint files) and computes the upper-triangle matrix of derivatives, parallelized over the available kernels. *)
(*        Moreover,  this function implements a checkpoint logic: *)
(*                              -If a checkpoint file exists: the computation was interrupted mid-run. Load the partially completed matrix and resume from where it left off. *)
(*                              - If no checkpoint but a smaller completed result exists: e.g., if a previous run computed N=150 and we now want N=200, it loads the 151*151 block and only computes the new rows (151 through 200). *)
(*                              - If starting fresh: initialize with zeros.*)
(*              *)
(*   3.  Matrix computation:  Each matrix is computed as follows: First, create lists of derivatives of F and G (or extend previous lists if the code has been ran for smaller N). Second, send these lists to the kernels in batches by using the function runBatchedLoop. Finally, recover the upper-triangle matrix and make use of the specific symmetry of the propagator to construct the whole matrix, which is saved as an .mx file.*)
(*              *)
(*        *)
(*Important notes:*)
(*- The matrices are saved as .mx files, which are not usable across versions of Mathematica. If the user runs this code with a specific version of Mathematica, then the next programs that make use of these .mx files must use the same version of Mathematica, otherwise the files will appear corrupted.*)
(*- This code has been tested on Dell R6525 nodes with AMD EPYC3 7763, 2.45 GHz cores on a HPC cluster, with 128 cores and 1024 GiB of RAM, up to N=200.*)
(**)


(* ::Subsection::Closed:: *)
(*1. Setup*)


(*Assumptions*)
$Assumptions = {T > 0, \[Sigma] > 0, \[CapitalOmega] > 0,a \[Element] Reals, b \[Element] Reals, L > 0};


(*Truncation N of the Hermite basis*)
Ntarget = 5;


(*Path to the directory where the files will be saved*)
targetDir = "replace/with/path/"; 
SetDirectory[targetDir];


(* ::Subsection::Closed:: *)
(*2. Functions*)


(* ::Subsubsection::Closed:: *)
(*Generators*)


(*H(\[Alpha], \[Beta])*)
GeneratorDeltaApBpTimePointlike[T_, \[CapitalOmega]_, L_, a_, b_] =
  -((E^((-L^2 + 2 (a - b) L T^2 + T^4 (a + b - 2 I \[CapitalOmega])^2)/(4 T^2)) +
     E^((-L^2 + 2 (-a + b) L T^2 + T^4 (a + b - 2 I \[CapitalOmega])^2)/(4 T^2))) T)/(4 L Sqrt[\[Pi]]);


(*W(\[Alpha], \[Beta])*)
GeneratorWmpTimePointlike[T_, \[CapitalOmega]_, a_, b_] =
  (E^((a^2 T^2)/2 + (b^2 T^2)/2 + I \[CapitalOmega] T^2 (a - b) - T^2 \[CapitalOmega]^2) -
   Sqrt[\[Pi]] E^(1/4 (a + b)^2 T^2) (\[CapitalOmega] - I (a - b)/2) T (1 -
      Erf[T \[CapitalOmega] - I T (a - b)/2]))/(4 \[Pi]);


(*\[CapitalDelta](\[Alpha], \[Beta])*)
GeneratorHApBpTimePointlike[T_, \[CapitalOmega]_, L_, a_, b_] =
  (T (E^((a^2 T^2 + b^2 T^2)/2 - T^2 (I (a + b) \[CapitalOmega] + \[CapitalOmega]^2) -
       (L - (a - b) T^2)^2/(4 T^2)) Erfi[(L - (a - b) T^2)/(2 T)] +
     E^((a^2 T^2 + b^2 T^2)/2 - T^2 (I (a + b) \[CapitalOmega] + \[CapitalOmega]^2) -
       (L + (a - b) T^2)^2/(4 T^2)) Erfi[(L + (a - b) T^2)/(2 T)]))/(4 L Sqrt[\[Pi]]);


(* ::Subsubsection::Closed:: *)
(*Helper*)


(*Core function*)

(*Arguments:
matName: identifier string ("DerivativesHTime", "DerivativesDeltaTime", or "DerivativesWmpTime")
dFL: list of derivatives of function F evaluated at 0: {F(0), F'(0), F''(0), ...}
dGL: list of derivatives of function G evaluated at 0: {G(0), G'(0), G''(0), ...}
hasSymFactor: Boolean \[LongDash] whether to include the factor (1+(-1)^(p+q)). This enforces a parity selection rule: terms where p+q is odd vanish (this factor equals 0 for odd p+q, and 2 for even p+q)
checkpointFile: filename for saving progress
checkpointInterval: how many rows to compute before saving a checkpoint*)

runBatchedLoop[matName_String, dFL_, dGL_,
               hasSymFactor_, checkpointFile_String,
               checkpointInterval_Integer] :=
  Module[{mat, lastRow, batchStart, batchEnd, batchResults, timeStart},
    timeStart = AbsoluteTime[];

    (* --- Initialise or resume from checkpoint --- *)
    If[FileExistsQ[checkpointFile],
      Print["  Found checkpoint for ", matName, ". Loading..."];
      {mat, lastRow} = Import[checkpointFile];
      Print["  Resuming from row ", lastRow + 1];
      ,

      (* Fresh start: build empty matrix, optionally seed from smaller saved block *)
      mat = ConstantArray[0, {Ntarget + 1, Ntarget + 1}];
      lastRow = -1;
      Module[{prevFiles, prevNs, prevN, oldMat},
        prevFiles = FileNames[matName <> "Aux*.mx"];
        prevNs = If[Length[prevFiles] > 0,
          Flatten[ToExpression[StringCases[#, DigitCharacter..]] & /@ prevFiles], {-1}];
        prevN = Max[prevNs];
        If[prevN >= 0 && prevN < Ntarget,
          Print["  Seeding from existing N=", prevN, " block..."];
          oldMat = Import[matName <> "Aux" <> ToString[prevN] <> ".mx"];
          mat[[1 ;; prevN + 1, 1 ;; prevN + 1]] = LowerTriangularize[oldMat];
          Clear[oldMat];
          lastRow = prevN;
        ];
      ];
    ];

    DistributeDefinitions[dFL, dGL, Ntarget, hasSymFactor];

    (* --- Process rows in batches for checkpointing --- *)
    batchStart = lastRow + 1;
    While[batchStart <= Ntarget,
      batchEnd = Min[batchStart + checkpointInterval - 1, Ntarget];
      Print["  ", matName, ": rows ", batchStart, "-", batchEnd,
            "  (", Round[100 batchEnd/(Ntarget + 1), 1], "%)",
            "  MemInUse: ", Round[MemoryInUse[]/1024^3, 0.01], " GB"];

      (* One task per row; CoarsestGrained keeps each kernel busy on a full row *)
      batchResults = ParallelTable[
        {i,
         Table[
           If[hasSymFactor,
             Sum[Binomial[i,q] Binomial[j,p] (-1)^p (1 + (-1)^(p+q))
                 dFL[[i+j-p-q+1]] dGL[[p+q+1]], (* = F^{(i+j-p-q)}(0) *)
                 {q, 0, i}, {p, 0, j}],
             Sum[Binomial[i,q] Binomial[j,p] (-1)^p
                 dFL[[i+j-p-q+1]] dGL[[p+q+1]],  (* = G^{(p+q)}(0) *)
                 {q, 0, i}, {p, 0, j}]
           ],
           {j, 0, i}]
        },
        {i, batchStart, batchEnd},
        Method -> "CoarsestGrained"
      ];

      (* Insert rows into matrix *)
      Do[
        mat[[batchResults[[k,1]] + 1, 1 ;; Length[batchResults[[k,2]]]]] =
          batchResults[[k,2]],
        {k, 1, Length[batchResults]}
      ];
      Clear[batchResults];

      Export[checkpointFile, {mat, batchEnd}];
      batchStart = batchEnd + 1;
    ];

    Print["  ", matName, " loop done. Time: ",
          Round[AbsoluteTime[] - timeStart, 0.1], " s"];
    mat
  ];


(* ::Subsection::Closed:: *)
(*3. Computation of the matrices*)


(* ::Subsubsection::Closed:: *)
(*Parallel Kernel setup*)


Quiet[CloseKernels[]];
LaunchKernels[];
Print["Launched ", $KernelCount, " parallel kernels."];


(* ::Subsubsection::Closed:: *)
(*Subscript[H, nm]*)


Block[{F, G, dFList, dGList, dFNew, dGNew, timeStart,
       oldMaxOrder, newMaxOrder, NlistPrev, checkpointFile,
       DerivativesHTimeProc, DerivativesHTimeAux, listFiles, existingListNs},

  checkpointFile = "DerivativesHTimeProc_checkpoint.mx";
  newMaxOrder    = 2 Ntarget;

  (* Skip if already done *)
  If[Max[Append[ToExpression[StringCases[#, DigitCharacter..]] & /@
        FileNames["DerivativesHTimeAux*.mx"] // Flatten, -1]] >= Ntarget,
    Print["H: N=", Ntarget, " already done. Skipping."];
    Return[];
  ];

  Print["\n=== Computing H derivatives ==="];
  timeStart = AbsoluteTime[];

  F[u_] = (T E^(1/4 T^2 (u - 2 I \[CapitalOmega])^2))/(4 L Sqrt[\[Pi]]);
  G[v_] = E^(-(L^2/(4 T^2))) E^(-((L v)/2)) Erfi[(L + v T^2)/(2 T)];

  (* Build or extend 1D derivative lists *)
  listFiles      = FileNames["dFListH*.mx"];
  existingListNs = Flatten[ToExpression[StringCases[#, DigitCharacter..]] & /@ listFiles];
  NlistPrev      = If[Length[existingListNs] > 0, Max[existingListNs], -1];
  oldMaxOrder    = 2 NlistPrev;

  If[NlistPrev >= 0,
    Print["H: Loading 1D lists from N=", NlistPrev, "..."];
    dFList = Import["dFListH" <> ToString[NlistPrev] <> ".mx"];
    dGList = Import["dGListH" <> ToString[NlistPrev] <> ".mx"];
    If[oldMaxOrder < newMaxOrder,
      Print["H: Extending 1D lists to order ", newMaxOrder, "..."];
      dFNew = Table[D[F[u], {u, k}] /. u -> 0, {k, oldMaxOrder+1, newMaxOrder}];
      dGNew = ParallelTable[
        D[G[v], {v, k}] /. v -> 0 /. Erfi[z_] -> 2/Sqrt[\[Pi]] E^z^2 DawsonF[z] // Simplify,
        {k, oldMaxOrder+1, newMaxOrder}, Method -> "CoarsestGrained"];
      dFList = Join[dFList, dFNew]; dGList = Join[dGList, dGNew];
      Clear[dFNew, dGNew];
    ];,
    Print["H: Computing 1D lists from scratch..."];
    dFList = Table[D[F[u], {u, k}] /. u -> 0, {k, 0, newMaxOrder}];
    dGList = ParallelTable[
      D[G[v], {v, k}] /. v -> 0 /. Erfi[z_] -> 2/Sqrt[\[Pi]] E^z^2 DawsonF[z] // Simplify,
      {k, 0, newMaxOrder}, Method -> "CoarsestGrained"];
  ];
  Export["dFListH" <> ToString[Ntarget] <> ".mx", dFList];
  Export["dGListH" <> ToString[Ntarget] <> ".mx", dGList];
  Print["H: 1D lists saved. MemInUse: ", Round[MemoryInUse[]/1024^3, 0.01], " GB"];

  DerivativesHTimeProc = runBatchedLoop[
    "DerivativesHTime", dFList, dGList, True, checkpointFile, 10];

  DerivativesHTimeAux = DerivativesHTimeProc +
    Transpose[DerivativesHTimeProc] -
    DiagonalMatrix[Diagonal[DerivativesHTimeProc]];
  Export["DerivativesHTimeAux" <> ToString[Ntarget] <> ".mx", DerivativesHTimeAux];
  Quiet[DeleteFile[checkpointFile]];
  Clear[DerivativesHTimeProc, DerivativesHTimeAux, dFList, dGList, F, G];
  Print["H total time: ", Round[AbsoluteTime[] - timeStart, 0.1], " s"];
];


(* ::Subsubsection::Closed:: *)
(*Subscript[\[CapitalDelta], nm]*)


Block[{F, G, dFList, dGList, dFNew, dGNew, timeStart,
       oldMaxOrder, newMaxOrder, NlistPrev, checkpointFile,
       DerivativesDeltaTimeProc, DerivativesDeltaTimeAux, listFiles, existingListNs},

  checkpointFile = "DerivativesDeltaTimeProc_checkpoint.mx";
  newMaxOrder    = 2 Ntarget;

  If[Max[Append[ToExpression[StringCases[#, DigitCharacter..]] & /@
        FileNames["DerivativesDeltaTimeAux*.mx"] // Flatten, -1]] >= Ntarget,
    Print["\[CapitalDelta]: N=", Ntarget, " already done. Skipping."];
    Return[];
  ];

  Print["\n=== Computing \[CapitalDelta] derivatives ==="];
  timeStart = AbsoluteTime[];

  F[u_] = (-T E^(-L^2/(4 T^2)) E^(1/4 T^2 (u - 2 I \[CapitalOmega])^2))/(4 L Sqrt[\[Pi]]);
  G[v_] = E^((v L)/2);

  listFiles      = FileNames["dFListD*.mx"];
  existingListNs = Flatten[ToExpression[StringCases[#, DigitCharacter..]] & /@ listFiles];
  NlistPrev      = If[Length[existingListNs] > 0, Max[existingListNs], -1];
  oldMaxOrder    = 2 NlistPrev;

  If[NlistPrev >= 0,
    Print["\[CapitalDelta]: Loading 1D lists from N=", NlistPrev, "..."];
    dFList = Import["dFListD" <> ToString[NlistPrev] <> ".mx"];
    dGList = Import["dGListD" <> ToString[NlistPrev] <> ".mx"];
    If[oldMaxOrder < newMaxOrder,
      Print["\[CapitalDelta]: Extending 1D lists to order ", newMaxOrder, "..."];
      dFNew = Table[D[F[u], {u, k}] /. u -> 0, {k, oldMaxOrder+1, newMaxOrder}];
      dGNew = ParallelTable[
        D[G[v], {v, k}] /. v -> 0 // Simplify,
        {k, oldMaxOrder+1, newMaxOrder}, Method -> "CoarsestGrained"];
      dFList = Join[dFList, dFNew]; dGList = Join[dGList, dGNew];
      Clear[dFNew, dGNew];
    ];,
    Print["\[CapitalDelta]: Computing 1D lists from scratch..."];
    dFList = Table[D[F[u], {u, k}] /. u -> 0, {k, 0, newMaxOrder}];
    dGList = Table[D[G[v], {v, k}] /. v -> 0 // Simplify, {k, 0, newMaxOrder}];
  ];
  Export["dFListD" <> ToString[Ntarget] <> ".mx", dFList];
  Export["dGListD" <> ToString[Ntarget] <> ".mx", dGList];
  Print["\[CapitalDelta]: 1D lists saved. MemInUse: ", Round[MemoryInUse[]/1024^3, 0.01], " GB"];

  DerivativesDeltaTimeProc = runBatchedLoop[
    "DerivativesDeltaTime", dFList, dGList, True, checkpointFile, 10];

  DerivativesDeltaTimeAux = DerivativesDeltaTimeProc +
    Transpose[DerivativesDeltaTimeProc] -
    DiagonalMatrix[Diagonal[DerivativesDeltaTimeProc]];
  Export["DerivativesDeltaTimeAux" <> ToString[Ntarget] <> ".mx", DerivativesDeltaTimeAux];
  Quiet[DeleteFile[checkpointFile]];
  Clear[DerivativesDeltaTimeProc, DerivativesDeltaTimeAux, dFList, dGList, F, G];
  Print["\[CapitalDelta] total time: ", Round[AbsoluteTime[] - timeStart, 0.1], " s"];
];


(* ::Subsubsection::Closed:: *)
(*Subscript[W, nm]*)


Block[{F, G, dFList, dGList, dFNew, dGNew, timeStart,
       oldMaxOrder, newMaxOrder, NlistPrev, checkpointFile,
       DerivativesWmpTimeProc, DerivativesWmpTimeAux, listFiles, existingListNs},

  checkpointFile = "DerivativesWmpTimeProc_checkpoint.mx";
  newMaxOrder    = 2 Ntarget;

  If[Max[Append[ToExpression[StringCases[#, DigitCharacter..]] & /@
        FileNames["DerivativesWmpTimeAux*.mx"] // Flatten, -1]] >= Ntarget,
    Print["W: N=", Ntarget, " already done. Skipping."];
    Return[];
  ];

  Print["\n=== Computing W derivatives ==="];
  timeStart = AbsoluteTime[];

  F[u_] = (E^(1/4 T^2 u^2))/(4 \[Pi]);
  G[v_] = E^(1/4 T^2 (v + 2 I \[CapitalOmega])^2) -
    Sqrt[\[Pi]] T (\[CapitalOmega] - I v/2) (1 - Erf[T (\[CapitalOmega] - I v/2)]);

  listFiles      = FileNames["dFListW*.mx"];
  existingListNs = Flatten[ToExpression[StringCases[#, DigitCharacter..]] & /@ listFiles];
  NlistPrev      = If[Length[existingListNs] > 0, Max[existingListNs], -1];
  oldMaxOrder    = 2 NlistPrev;

  If[NlistPrev >= 0,
    Print["W: Loading 1D lists from N=", NlistPrev, "..."];
    dFList = Import["dFListW" <> ToString[NlistPrev] <> ".mx"];
    dGList = Import["dGListW" <> ToString[NlistPrev] <> ".mx"];
    If[oldMaxOrder < newMaxOrder,
      Print["W: Extending 1D lists to order ", newMaxOrder, "..."];
      dFNew = Table[D[F[u], {u, k}] /. u -> 0, {k, oldMaxOrder+1, newMaxOrder}];
      dGNew = ParallelTable[
        D[G[v], {v, k}] /. v -> 0 // Simplify,
        {k, oldMaxOrder+1, newMaxOrder}, Method -> "CoarsestGrained"];
      dFList = Join[dFList, dFNew]; dGList = Join[dGList, dGNew];
      Clear[dFNew, dGNew];
    ];,
    Print["W: Computing 1D lists from scratch..."];
    dFList = ParallelTable[D[F[u], {u, k}] /. u -> 0 // Simplify,
      {k, 0, newMaxOrder}, Method -> "CoarsestGrained"];
    dGList = ParallelTable[D[G[v], {v, k}] /. v -> 0 // Simplify,
      {k, 0, newMaxOrder}, Method -> "CoarsestGrained"];
  ];
  Export["dFListW" <> ToString[Ntarget] <> ".mx", dFList];
  Export["dGListW" <> ToString[Ntarget] <> ".mx", dGList];
  Print["W: 1D lists saved. MemInUse: ", Round[MemoryInUse[]/1024^3, 0.01], " GB"];

  DerivativesWmpTimeProc = runBatchedLoop[
    "DerivativesWmpTime", dFList, dGList, False, checkpointFile, 10];

  (* W uses ConjugateTranspose, not plain Transpose *)
  DerivativesWmpTimeAux = DerivativesWmpTimeProc +
    ConjugateTranspose[DerivativesWmpTimeProc] -
    DiagonalMatrix[Diagonal[DerivativesWmpTimeProc]];
  Export["DerivativesWmpTimeAux" <> ToString[Ntarget] <> ".mx", DerivativesWmpTimeAux];
  Quiet[DeleteFile[checkpointFile]];
  Clear[DerivativesWmpTimeProc, DerivativesWmpTimeAux, dFList, dGList, F, G];
  Print["W total time: ", Round[AbsoluteTime[] - timeStart, 0.1], " s"];
];


(* ::Subsection::Closed:: *)
(*5. Clean-up*)


CloseKernels[];
Print["\nFinished."];
