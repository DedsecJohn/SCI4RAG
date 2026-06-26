"""
Stage-2 feature-based adjudication for citation-edge classification.

Stage-1 (LLM citation-context intent) labels each directed edge as
``inheritance`` or ``unknown``. Stage-2 consumes the *raw* per-edge feature
vector (never aggregated into a single scalar) to adjudicate the ``unknown``
edges into the final label set ``{inheritance, parallel, peripheral}``:

- Subtask A (false-negative recovery): asymmetric family dominant
  (``citation_freq`` / ``pagerank_target``) recovers mislabeled inheritance.
- Subtask B (parallel vs peripheral): symmetric family dominant
  (``cocitation_salton`` / ``bibcoupling_jaccard`` / ``author_jaccard`` with
  content similarity as a confidence booster).
"""
