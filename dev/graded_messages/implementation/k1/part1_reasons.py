"""Write k1/full_weight_reasons.tsv for part 1 from the audit's NOREASON lines and a hand-written reason per site."""
import re, sys
audit = open(sys.argv[1]).read().splitlines()
R = {
 # the reference's docstrings (prose_edits.md section 6)
 ('doc:w_tilde:<module>', 'block=block'): ('probe', 'the example\'s per-mode-and-slot weight, (d,)+F: today (3,3,1,1) with size-1 sample and edge, the same nine draws'),
 ('doc:xi_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:mu_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:nu_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:eta_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:nu_L_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:eta_D_tilde:<module>', 'block=0.5 + rng.random(3)'): ('probe', 'one weight per mode, (d,) at order (): the old (3,1,1) per-mode weight, the same three draws'),
 ('doc:f:<module>', 'block=per_mode'): ('probe', 'one value per mode, (d,) at order (): the block form of the old per_mode.reshape(3,1,1)'),
 ('doc:r_block_sumsq:<module>', 'block=block'): ('probe', 'a new example (prose 6.6): a block weight scales its block'),
 ('doc:JtJp:<module>', 'full=full'): ('probe', 'the old per-mode-and-sample weight (3,7,1) written out on the edge (np.broadcast_to): a full weight, the same values'),
 ('doc:f:<module>', 'full=full'): ('probe', 'one per entry, z\'s shape (the old weight, renamed)'),
 ('doc:f:<module>', 'full=per_mode.reshape(3, 1, 1)'): ('probe', 'the refusal shown: a per-mode value passed as a full weight with size-1 sample and edge axes raises'),
 ('doc:r_block_sumsq:<module>', 'full=full'): ('probe', 'one per sample, written out on the slots and the edge (ones with a zero sample): the old (1,7,1) form'),
 ('doc:Jp_norm_sq:<module>', 'full=full'): ('probe', 'the old per-mode-and-sample weight (3,7,1) written out on the edge: a full weight, the same values'),
 # tests
 ('TestByDefinition.check', 'block=block'): ('apply', 'problem.block_weight: the blocks F whose blocks_to_weight placement the engine gets (problem.data); the old reference weight was that placement'),
 ('reference', 'block=block'): ('probe', 'problem.block_weight: the blocks (d,)+F whose blocks_to_weight placement the engine gets (problem.data)'),
 ('TestJetsTwoWays.test_a_second_variable_of_order_zero_is_one_variable', 'block=weight'): ('probe/apply', 'the block table of Problem.arguments with a slot axis of 1 appended for the second jet variable of order 0 (F = (4, 1))'),
 ('TestKp.jax_problem', "block=jnp.asarray(args['block_weight'])"): ('all three', 'Problem.arguments\' block weight on jax'),
 ('Problem.arguments', 'block=self.block_weight(kind, case)'): ('all three', 'the blocks of Problem.blocks; the old value, full_weight(), was blocks_to_weight of the same blocks'),
 ('TestTheTwoWeights.test_against_brute_force', 'block=block'): ('all three', 'the brute-force pin\'s block weight, a random table (d,)+F / F / a number'),
 ('TestTheTwoWeights.test_against_brute_force', 'full=f_arg'): ('all three', 'the brute-force pin\'s full weight: per entry, per slot written out, with W of 1 (the patterns)'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'block=weight'): ('all three', 'the refusal table of block weights of a wrong shape'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'block=list(np.atleast_1d(blocks))'): ('all three', 'the refusal of a sequence (TypeError)'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'block=blocks'): ('all three', 'the right form passes: the blocks of Problem.blocks'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'block=np.ones(P.d)'): ('probe and apply', 'the E3b coincidence: refused for probe, legal (per slot) for apply'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'full=weight'): ('all three', 'the refusal table of full weights with a structural axis of 1 or another form'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'full=of_kind(full)'): ('all three', 'the right form passes: per entry, z\'s shape'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'full=tuple(full)'): ('probe', 'per entry: a tuple and a list are one object'),
 ('TestTheTwoWeights.test_a_weight_of_another_form_raises', 'full=full'): ('probe', 'per entry: a tuple and a list are one object'),
 ('TestTheTwoWeights.test_an_object_larger_than_z_where_z_has_a_size_of_one_raises', 'full=good'): ('probe and apply', 'the old weight=good: ones of z\'s shape, per entry'),
 ('TestTheTwoWeights.test_an_object_larger_than_z_where_z_has_a_size_of_one_raises', 'full=bad'): ('probe and apply', 'the old weight=bad: larger than z where z is 1, refused'),
 ('TestByDefinition.test_every_name', 'block=block'): ('entries', 'problem.block_weight: the one number whose blocks_to_weight placement the engine gets (problem.data)'),
 ('TestTheTwoWeights.test_every_name_that_reads_the_seed_reads_both_weights', 'block=block'): ('all three', 'the law test: a random block table, the same one written into the seed by loops'),
 ('TestTheTwoWeights.test_every_name_that_reads_the_seed_reads_both_weights', 'full=full'): ('all three', 'the law test: a full weight with W of 1, the same one written into the seed by loops'),
 ('TestAgainstTheReference.weights', 'full=None if full is None else of_kind(kind, full)'): ('all three', 'M2\'s patterns per entry, per sample and on the edge and first jet axis (written out), with W of 1, and the full half of both; the engine gets the same array (or its product with blocks_to_weight of the block)'),
 ('TestTheReferenceAnchor.test_against_the_reference', 'full=weight_c'): ('probe', 'case.weight is one per entry, box+W+C+(N_i,), element c of C: z\'s shape exactly'),
}
out = ['# K1 part 1: the reason of every new full_weight= (and every new block_weight= that is not a blocks_to_weight',
       '# table of its scope), one line per site: file, scope, callee, keyword, expression, kind, reason.']
rx = re.compile(r'^([BF])-NOREASON (\S+?):\d+ (\S+): (.*)\((block_weight|full_weight)=(.*)\), first argument .*$')
missing = 0
for line in audit:
    mm = rx.match(line)
    if not mm:
        continue
    bf, path, scope, callee, kw, expr = mm.groups()
    key = (scope, ('block=' if kw == 'block_weight' else 'full=') + expr)
    if key not in R:
        print('NO REASON WRITTEN', line); missing += 1; continue
    kind, why = R[key]
    out.append('\t'.join([path, scope, callee, kw, expr, kind, why]))
open(sys.argv[2], 'w').write('\n'.join(out) + '\n')
print(len(out) - 2, 'lines;', missing, 'missing')
