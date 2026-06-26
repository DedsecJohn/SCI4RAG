"""
Citation feature engineering (FeatureForge).

Builds quantitative citation-relatedness features for the papers of a target
``{username}/{dataset}`` directory, as inputs for downstream citation-importance
modelling. See the requirements document for the authoritative definitions.

The candidate universe is restricted to *directed citation edges* between local
papers (those with cleaned ``doi.json`` / ``references.json`` / ``document.md``
under ``data_clean``); each row is "paper i cites paper j":

- Bibliographic coupling  -> local ``references.json`` reference lists.
- Co-citation             -> ``citation_network.json`` in-edges (whole graph).
- PageRank                -> ``citation_network.json`` directed graph (whole graph).
- Author Jaccard          -> local ``doi.json`` authors.
- Semantic similarity     -> local ``doi.json`` title / abstract (SPECTER2), split
                             into ``title_sim`` and ``abstract_sim``.
- Citation frequency (CF) -> local ``citation_by_article.json`` in-text counts;
                             directional ``citation_freq = N_ij / T_i``.
"""

from src.service.gragh.citation.config import CitationFeatureConfig

__all__ = ["CitationFeatureConfig", "run"]

__version__ = "3.0.0"


def __getattr__(name):
    # Lazy import to avoid pulling heavy/optional deps (and circulars) on
    # ``import src.service.gragh.citation``.
    if name == "run":
        from src.service.gragh.citation.pipeline import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
