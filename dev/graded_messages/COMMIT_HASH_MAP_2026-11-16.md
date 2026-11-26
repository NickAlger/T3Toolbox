# Commit hashes before and after the author rewrite of 2026-11-16

On 2026-11-16 the commits made by Claude in this arc were rewritten so that NickAlger is their author and
committer, with Claude in the co-author line, as on the rest of `main` (the maintainer's instructions of 17:07,
17:09 and 17:17). That is the last two commits of `main` (the working folder and the handoff note) and the 77
commits then on the branch `graded-messages`. Only the author and committer changed: the trees, the dates and the
messages are the same. Every hash changed, and both `main` and the branch were force-pushed.

The hashes cited in this folder give the new values, and the four files named after a hash were renamed. Four
commit messages cite an earlier hash and were left as written. This table resolves them, and any old hash met
elsewhere (a doc tab, a note outside the repository). The 77 were rewritten twice, first alone and then again
when `main` moved under them; the middle column is the value they had between the two, for about ten minutes.

| # | on | original | between | now | subject |
|---|---|---|---|---|---|
| 1 | main | `c7f77f6` | (same as original) | `baa3090` | dev: graded-messages working folder -- the design work for the sampling-layer refactor moves into the repository |
| 2 | main | `d4c2cf8` | (same as original) | `c96b19d` | dev: handoff -- the graded-messages design work is tracked in dev/graded_messages/ and continues on the branch graded-messages |
| 3 | branch | `860d143` | `4731def` | `f6c59b1` | dev: graded messages -- the direction and coordinate rescalings are library functions (6.7); the adapter proposal (2.3); the later plan confirmed |
| 4 | branch | `be509da` | `d8e2da9` | `fbc7d44` | dev: graded messages -- the derivative-mode adapter decided (2.3), with a draft of its doctest |
| 5 | branch | `a6746c1` | `e06094f` | `5ba6411` | dev: graded messages -- the documents for decision 2.3 (the previous commit held only the doctest draft) |
| 6 | branch | `bb36121` | `a1724a9` | `27b5c76` | dev: graded messages -- how a term is carried out over the orders is decided (2.9): a scan over a full operand's coefficients |
| 7 | branch | `f217a1b` | `111d7e4` | `0b34cbf` | dev: graded messages -- the scan of 2.9 is being reconsidered; three forms of the product of two full jets compared |
| 8 | branch | `c9865c1` | `4377301` | `ff98fd4` | dev: graded messages -- how the library carries out a product of two full jets today, and a corrected comparison (2.9) |
| 9 | branch | `a6f8abd` | `e3bfe66` | `3c2ec93` | dev: graded messages -- 2.9 decided: two forms over the orders, chosen through the plan; the scan is the default |
| 10 | branch | `68daa4e` | `ae64508` | `adadf5a` | dev: graded messages -- the scan of 2.9 is written through xscan; Nick's rules for dispatch recorded |
| 11 | branch | `ee23f44` | `9a08850` | `d8a2b62` | dev: graded messages -- the library's own patterns recorded beside Nick's rules; the scan prototype follows them |
| 12 | branch | `794bc1c` | `ed27603` | `1ea27e7` | dev: graded messages -- what "public" means for a variable: Nick's take and my answer, under discussion |
| 13 | branch | `30a3b09` | `f6c3e7a` | `3c31c6b` | dev: graded messages -- public: Nick's two objections and my revised answer (provide every variable) |
| 14 | branch | `e56120f` | `d2c8cef` | `ab512df` | dev: graded messages -- public confirmed (every variable; affine two-slot group); the six variables without names |
| 15 | branch | `2fc84f6` | `c3010b1` | `8a1d1a6` | dev: graded messages -- zeta decided; type catalog section 13: the letters by edge, and the dot collision |
| 16 | branch | `3271be3` | `475b84a` | `6085f12` | dev: graded messages -- type catalog section 14: a variable is a contracted subnetwork in a basis; one dot |
| 17 | branch | `6e0910b` | `005b954` | `a4340be` | dev: graded messages -- the four-letters-and-a-basis scheme written out against every variable |
| 18 | branch | `d90f390` | `5f5b187` | `1054dfb` | dev: graded messages -- prototype of a table of diagrams, one row per variable |
| 19 | branch | `772f2f4` | `a4971a6` | `fa8054d` | dev: graded messages -- variable diagrams: Greek letters, tildes, dots and subscripts inside the drawings |
| 20 | branch | `fe5fbf7` | `cad5cbb` | `5d15a47` | dev: graded messages -- the notation is decided (four letters and a basis); the sample variables page and its check |
| 21 | branch | `1b3c5db` | `bed628f` | `5a13d48` | dev: graded messages -- the remaining 21 variable diagrams, drawn from the page alone and checked; the survey for the notation change |
| 22 | branch | `e2e8fc6` | `8222abe` | `474be90` | dev: graded messages -- the tilde variables are the library's adjoint variables: checked array by array, and the page says so |
| 23 | branch | `e05a0dc` | `ab77fab` | `18a956a` | dev: graded messages -- variable diagrams: the tilde rows say that the variable is one array, the sum over the modes |
| 24 | branch | `70073af` | `e1213c7` | `38a6bd8` | dev: graded messages -- the page: step functions are named by letter and grade and serve both bases; the corewise principle was set aside for its sweep cost |
| 25 | branch | `1dd65d6` | `27fb13a` | `78b4016` | dev: graded messages -- notation change, editor round 1: the folder brought into line with SAMPLE_VARIABLES.md (unreviewed) |
| 26 | branch | `621b9a1` | `f183e46` | `6700db5` | dev: graded messages -- the raw core direction is (D_dot, L_dot); step functions by letter and grade; the corewise principle set aside (log entry 59) |
| 27 | branch | `ed574ac` | `8f6d697` | `edc584b` | dev: graded messages -- notation change: the reviewer's report on round 1; the page's name map gains the raw core direction |
| 28 | branch | `e073074` | `dbe35d6` | `f478555` | dev: graded messages -- notation change, editor round 2: the reviewer's findings fixed; the raw core direction is (D_dot, L_dot) in the folder's documents and scripts (under review) |
| 29 | branch | `5ca2030` | `02d5a05` | `da42eb2` | dev: graded messages -- 3.6 decided: the engine takes y and an optional weight, applied through one public helper; f, g, JtJp and Kp share it (log entry 60) |
| 30 | branch | `9b89f0d` | `c8ded56` | `f67b1dc` | dev: graded messages -- notation change: the reviewer is satisfied with the folder (round 2); three wording points fixed |
| 31 | branch | `d17b68c` | `271f278` | `312a068` | dev: graded messages -- the sample stack on the outputs: transposes sum by default, the sampling layer supports C and K, the fitting layer may assume none (log entry 61) |
| 32 | branch | `73e42f0` | `234e7b6` | `4f9481f` | dev: graded messages -- notation change: the editor's record of bringing the doc tabs in step (under review) |
| 33 | branch | `fe62de9` | `77c3a9c` | `b3a7d25` | dev: graded messages -- 3.7 decided: what happens to the sample stack on each output (log entry 62) |
| 34 | branch | `24e5174` | `2c356cc` | `0d23d99` | dev: graded messages -- 3.3 decided (the tilde names mean the residual's adjoint sweep); the notation change closed: doc review, README, log entries 63 and 64 |
| 35 | branch | `11b0d9b` | `14c1425` | `a805b19` | dev: graded messages -- the editor's record of the last tab sync (3.3, 3.6 wording, 3.7, the sample stack note) |
| 36 | branch | `1df3fa6` | `4023904` | `f4ed79c` | dev: graded messages -- 4.1 decided: the frame stays four stores, optional Tucker centers give the value off the frame, wrong centers error in safe mode (log entry 65) |
| 37 | branch | `df17105` | `7268d54` | `235a045` | dev: graded messages -- the center cores are Vc and Hc; the page has the value's second route, and its check compares the three routes on the library's frame |
| 38 | branch | `b2ac945` | `a68afbf` | `d4349ab` | dev: graded messages -- the editor's record of the tab sync for 4.1 and Vc, Hc |
| 39 | branch | `fd733d1` | `2744899` | `0da7142` | dev: graded messages -- 4.3 decided: three policies for a trial point as options of the optimizers, a frame at every trial by default (log entry 66) |
| 40 | branch | `30cdcfd` | `6a814e8` | `70ef860` | dev: graded messages -- 4.2 decided: frame consistency is a safe-mode precondition, checked only where needed and never through a dense tensor (log entry 67) |
| 41 | branch | `cea4c1a` | `ce25c90` | `0e1ab86` | dev: graded messages -- 4.4 decided: frames support arbitrary tails, unchecked; an audit of functions that require squashed tails noted for the branch (log entry 68) |
| 42 | branch | `d4025e4` | `d961c38` | `e2e07d0` | dev: graded messages -- group B finished: 3.2 and 4.5 decided; the tails discipline limited to the functions the refactor touches (log entry 69) |
| 43 | branch | `882a9cc` | `4682cd8` | `ce0095f` | dev: graded messages -- 3.4 decided: a kind is closed in the sampling layer; the open extension point is an objective protocol in the fitting layer (log entry 70) |
| 44 | branch | `4392a42` | `6868cd9` | `6171acd` | dev: graded messages -- forward-only probing left off this arc |
| 45 | branch | `5e5b8b9` | `5f9d1aa` | `249ba05` | dev: graded messages -- 6.5 decided: held variables are trusted, shapes are checked (log entry 71) |
| 46 | branch | `d0bca98` | `60bfc1a` | `8187040` | dev: graded messages -- fix the rows 3.3 and 6.5 of the agenda (a misplaced replacement in the last commit) |
| 47 | branch | `254e7d5` | `5f61825` | `45ae412` | dev: graded messages -- 6.2 decided up to three names: the arguments of sampling_variables (log entry 72) |
| 48 | branch | `20df9aa` | `367f1c8` | `3ea0b84` | dev: graded messages -- 6.4 decided: the table of variables is written out and tested against the declared graph (log entry 73) |
| 49 | branch | `9ccfca2` | `eaaf4a6` | `69ce8c5` | dev: graded messages -- 6.3 decided: the public functions of the sampling layer (log entry 74) |
| 50 | branch | `fc436b2` | `bd097e0` | `0bc6b64` | dev: graded messages -- 8.2 decided, the name apart: the fitting layer's extension point is one operation in the engine's vocabulary (log entry 75) |
| 51 | branch | `9feb79e` | `bc6e93a` | `726a755` | dev: graded messages -- the fitting layer's extension point is called an oracle |
| 52 | branch | `67af17f` | `0c88b86` | `3bd1696` | dev: graded messages -- group C finished: 9.2 decided (three kind strings; hold lists variable names; sizing names moved to 7.6) (log entry 76) |
| 53 | branch | `fddb3d4` | `25fd511` | `06d4181` | dev: graded messages -- 5.2 decided: one set of step functions for both layouts (log entry 77) |
| 54 | branch | `3dc373b` | `50babbd` | `ebf17ee` | dev: graded messages -- 5.6 decided (absorbing dtype rule); 5.5 reversed: the contraction order goes into the plan; how the library places K in a term (log entry 78) |
| 55 | branch | `ef1828f` | `149633e` | `e278ea7` | dev: graded messages -- 5.5: one contraction order per term function; helpers are a design phase; how an order reads with jets (log entry 79) |
| 56 | branch | `2b609bd` | `2ee26ff` | `9f6ac57` | dev: graded messages -- 5.5: the granularity of the contraction order moves to the performance discussion (log entry 80) |
| 57 | branch | `e4e6448` | `c25e8be` | `acc209c` | dev: graded messages -- record of bringing the Target decisions tab in step with the file (to 2b609bd) |
| 58 | branch | `742a89c` | `02d51aa` | `710bf51` | dev: graded messages -- 8.1 under discussion: what LocalModel keeps; the weighted quadratic checked against <p, JtJp> (log entry 81) |
| 59 | branch | `8b86c0c` | `bebe1e8` | `229da9d` | dev: graded messages -- 8.1 decided (what LocalModel is); Jp_sumsq renamed Jp_norm_sq; the pause of 2026-11-16, 04:46 (log entry 82) |
| 60 | branch | `4189c93` | `91b7c95` | `d26b272` | dev: graded messages -- the Claude doc brought in step at the pause; a pointer in the 8.1 note corrected |
| 61 | branch | `ad7e0ed` | `dcda1d5` | `3b97108` | dev: graded messages -- the tab's revision in the sync record corrected (166) |
| 62 | branch | `33095c5` | `7f67581` | `4cf38d7` | dev: graded messages -- 8.3 part 1 decided: the optimizers' contract; oracle.evaluate with want and held (log entry 83) |
| 63 | branch | `d092215` | `a8bba82` | `0ba9d41` | dev: graded messages -- 8.3 part 2 decided: a minibatch is an oracle, scaled to estimate the full one (log entry 84) |
| 64 | branch | `316b3a4` | `24af742` | `1065955` | dev: graded messages -- 8.3 decided: five phases as the units of compilation, the host keeps the decisions (log entry 85) |
| 65 | branch | `153edb2` | `c979b9d` | `6a08676` | dev: graded messages -- the Target decisions tab brought in step with 316b3a4 (record of the sync and its check) |
| 66 | branch | `a2f66c5` | `6198140` | `32c8194` | dev: graded messages -- 8.4 in part: a validation set is an oracle, the floor is the oracle's, no table without block sums; the derivative-mode point open |
| 67 | branch | `0573bca` | `679f254` | `25d818c` | dev: graded messages -- 8.4 decided; derivative mode moved out of the fitting layer, 2.2 and 2.3 amended (log entry 86) |
| 68 | branch | `7afe516` | `dd6be86` | `2fd9d05` | dev: graded messages -- the oracle is a description and the data are an argument; 8.1 to 8.4 reworded, 6.7 amended (log entry 87) |
| 69 | branch | `49e3d1e` | `d986c23` | `33a42eb` | dev: graded messages -- the Target decisions tab brought in step with 7afe516; an older difference in the 6.7 notes recorded |
| 70 | branch | `c3c3bee` | `533f2e5` | `c7d751e` | dev: graded messages -- audit of the Target decisions tab against the file: what was fixed and what remains |
| 71 | branch | `72ec6e0` | `14ce911` | `fa1b11d` | dev: graded messages -- 9.3 decided: correct is defined by the definitions (log entry 88) |
| 72 | branch | `6e8264a` | `1892521` | `eadbf74` | dev: graded messages -- 9.7 added and decided: how the sampling variables are documented (log entry 89) |
| 73 | branch | `6b24f8a` | `8b74d3c` | `8f1d2bd` | dev: graded messages -- 5.4: prototype of the seven terms generic over every stack and the order (log entry 90) |
| 74 | branch | `de1207b` | `ce47938` | `e006d90` | dev: graded messages -- 5.4: a literal contraction string for each place the direction sits (log entry 91) |
| 75 | branch | `5581694` | `f67dc9c` | `bee2f04` | dev: graded messages -- 5.4 decided: literal strings at the call sites through jet_contract; the prototype rewritten (log entry 92) |
| 76 | branch | `0b55de4` | `a3cf5e3` | `67c41b5` | dev: graded messages -- 5.4: the lengths go to every call as one record (log entry 93) |
| 77 | branch | `5ffc3d1` | `6a4101d` | `652b975` | dev: graded messages -- the refactor split into two parts: what is computed, and how it is carried out (log entry 94) |
| 78 | branch | `74fbeaa` | `c960188` | `e4894d4` | dev: graded messages -- the small items of Part 1 settled (log entry 95) |
| 79 | branch | `14aa727` | `b1d8c00` | `4d38df7` | dev: graded messages -- the brief for the implementation plan of Part 1 |
| 80 | branch | (made after the first rewrite) | `e26fb3c` | `be0248a` | dev: graded messages -- cited hashes updated after the author rewrite; the old-to-new table |
| 81 | branch | (made after the first rewrite) | `3b54276` | `9f42cab` | dev: graded messages -- the implementation plan of Part 1, draft r1 (the writer's, not yet reviewed) |
