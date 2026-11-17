# Which pairs to match

A capture of n images has n (n - 1) / 2 pairs, and most of them show different places. Matching a pair at full
settings costs 100 to 500 ms on a GPU, so a few hundred images cannot all be matched with each other. just-matching
scores pairs at three levels of cost. Each level uses only what the steps before it computed, so no second
network is needed.

| Level | What it is | Cost per pair | Use |
|---|---|---|---|
| 1 | Cosine similarity of the two images' mean feature vectors | a dot product | shortlist the partners of each image |
| 2 | A 16 x 16 grid of points of each image through FlexGlue's first layer to the other image and back; the share of points that come back within one cell | about 0.3 G multiply-adds | re-rank the shortlist |
| 3 | FlexGlue's own covisibility score, after each transformer layer | 25 to 100 % of a match | the verdict, while matching |

## The shortlist: levels 1 and 2

```c
jmt_shortlist_options so;
jmt_shortlist_options_default(&so);
so.k = 10;                 /* partners kept per image by level 1; a pair is kept when either image keeps it */
so.neighbours = 2;         /* images given in capture order: also keep (i, i + 1) and (i, i + 2) */
so.rerank = 1;             /* score the kept pairs with level 2 and sort by that score */
so.min_score = 0;          /* with rerank: drop pairs below this level-2 score */

jmt_pair_list pairs;
jmt_shortlist(ctx, features, n_images, &so, &pairs);
for (int i = 0; i < pairs.n; ++i) {
  int a = pairs.idx[2 * i], b = pairs.idx[2 * i + 1];   /* a < b, best pairs first */
  float score = pairs.score[i];
}
```

`features` is the array of `jmt_features*` from `jmt_describe`. Each image's mean vector (for level 1) and a
64-channel map (for level 2, 1 MB in FP32) are computed on first use and kept with its features. One pair at one
level: `jmt_covisibility(ctx, fa, fb, level, &score)`.

From the command line:

```bash
jmt shortlist images/*.jpg --k 10 --neighbours 2 --rerank --out pairs.jmtt
```

| Option of `shortlist` | Meaning |
|---|---|
| `--k K` | partners kept per image by level 1 (default 10; n - 1 or more keeps every pair) |
| `--neighbours N` | also keep the N next images in the order given (default 0) |
| `--rerank` | score the kept pairs with level 2 and sort by it |
| `--min-score S` | with `--rerank`, drop pairs scoring below S (default 0) |
| `--backend`, `--weights`, `--size`, `--threads` | as for `match` |

### How good the scores are

On 53 images (frames of an indoor and a street sequence, photos of two landmarks and three single images; 1378
pairs, 400 of them in the same scene), all three levels separate the scenes. The areas under the ROC curve for "same scene" are 0.9997 for
level 1, 0.9998 for level 2 and 1.0000 for level 3. With k = 5 the shortlist keeps 156 pairs of the 1378.

- **Level 1 scores sit in a narrow range**, 0.8 to 1 in practice: compare pairs with each other, not with a fixed
  bar. That is why the shortlist keeps the best k partners per image rather than every pair above a score.
- **Level 1 ranks by look more than by place.** In the demo's 12 sample images, two views of one place in
  Berlin, one of them a close-up, rank 7th of 11 for each other, behind the four street frames and the two LaMAR
  images. With images in capture order, `neighbours` keeps such pairs.
- **Level 2 needs the same scale.** The Berlin pair scores 0.11 at level 2, against 0.99 from FlexGlue itself. The
  grid looks for each point in one cell of the other image, and a close-up spreads one cell over many.
- Ranking partial overlap inside one capture, which is the shortlist's real job, has not been measured against a
  reference such as MegaLoc yet.

## The verdict: level 3 and early stop

FlexGlue predicts, after each layer, the probability that the two images show the same place. The original
rejects a pair scoring below a bar and stops early on pairs that fall well below it. just-matching applies the
same rule:

```c
jmt_match_options mo;
jmt_match_options_default(&mo);
mo.min_covisibility = 0.8f;    /* pairs below it get no matches */
mo.early_stop_decay = 0.2f;    /* and stop after a few layers */
jmt_match(ctx, fa, xy_a, n_a, fb, xy_b, n_b, &mo, &m);
/* m.covisibility: the score; m.stop_layer < 9 when the pair stopped early */
```

After layer k (from 2 to 8), a pair stops when its score is below `min_covisibility * (1 - decay)^(9 - k)`. It
then gets no matches, but its predicted points and confidences are still returned. A pair that passes runs on to
full depth and is matched in the same call: nothing is computed twice. Each check costs 1 to 3 % of a pair that
passes. A pair stopped after layer 2 costs about a quarter of a full one.

| Decay | Pairs of other scenes stopped | Mean stop layer | Transformer work left | Same-scene pairs lost |
|---|---|---|---|---|
| 0 | 99.8 % | 2.0 | 45 % | 1, and 18 others that pass at layer 9 |
| 0.1 | 95.4 % | 3.7 | 60 % | none (1 other) |
| 0.2 | 84.9 % | 5.6 | 77 % | none |
| 0.5 | 41.9 % | 7.9 | 96 % | none |

Every pair of the 53 images, with `min_covisibility` 0.8. "Lost" counts pairs that would score 0.8 or more after
9 layers but stop early. 0.2 is what the original's README uses; 0.5 is gentler.

- **Unrelated scenes score 0.3 to 0.7** at full depth, not near zero; look-alikes of different places score near
  zero. Hence the high bar: 0.8, as the original suggests.
- **After a shortlist there is little left to stop.** On the k = 5 shortlist of the same images, decay 0.2 stops
  16 pairs and saves 6 % of the transformer's work. Early stop pays most when matching every pair of a small
  capture, and on look-alikes that the shortlist lets through.
- **A pair can pass with a handful of matches.** In the demo's samples, a KITTI street frame and a LaMAR image of
  another place score 0.99 with 16 matches. A minimum number of matches is a useful second test.
