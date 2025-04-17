from argdown_feedback.tasks.core import (
    arganno,
    argmap,
    infreco,
    logreco,
)
from argdown_feedback.tasks.sequential import (
    arganno_from_argmap,
    argmap_from_arganno,
    infreco_from_arganno,
    logreco_from_infreco,
)
from argdown_feedback.tasks.compound import (
    arganno_plus_infreco,
    arganno_plus_logreco,
    argmap_plus_arganno,
    argmap_plus_infreco,
    argmap_plus_logreco,
    argmap_plus_arganno_plus_logreco,
)


_CLASS_REGISTRY = {
    # module arganno
    "arganno.Annotation": arganno.Annotation,
    "arganno.AnnotationProblem": arganno.AnnotationProblem,
    "arganno.AnnotationProblemGenerator": arganno.AnnotationProblemGenerator,
    "arganno.AnnotationJudge": arganno.AnnotationJudge,
    "arganno.AnnotationFeedbackGenerator": arganno.AnnotationFeedbackGenerator,
    "arganno.AnnotationAttacksPreferencePairGenerator": arganno.AnnotationAttacksPreferencePairGenerator,
    "arganno.AnnotationCoveragePreferencePairGenerator": arganno.AnnotationCoveragePreferencePairGenerator,
    "arganno.AnnotationNoAttacksPreferencePairGenerator": arganno.AnnotationNoAttacksPreferencePairGenerator,
    "arganno.AnnotationScopePreferencePairGenerator": arganno.AnnotationScopePreferencePairGenerator,
    "arganno.AnnotationSupportsPreferencePairGenerator": arganno.AnnotationSupportsPreferencePairGenerator,
    # module argmap
    "argmap.ArgumentMap": argmap.ArgumentMap,
    "argmap.ArgMapProblem": argmap.ArgMapProblem,
    "argmap.ArgMapProblemGenerator": argmap.ArgMapProblemGenerator,
    "argmap.ArgMapJudge": argmap.ArgMapJudge,
    "argmap.ArgMapFeedbackGenerator": argmap.ArgMapFeedbackGenerator,
    "argmap.BalancePreferencePairGenerator": argmap.BalancePreferencePairGenerator,
    "argmap.DensityPreferencePairGenerator": argmap.DensityPreferencePairGenerator,
    "argmap.ConnectednessPreferencePairGenerator": argmap.ConnectednessPreferencePairGenerator,
    "argmap.MaxDiameterPreferencePairGenerator": argmap.MaxDiameterPreferencePairGenerator,
    "argmap.MinDiameterPreferencePairGenerator": argmap.MinDiameterPreferencePairGenerator,
    "argmap.MaxInDegreePreferencePairGenerator": argmap.MaxInDegreePreferencePairGenerator,
    "argmap.MaxOutDegreePreferencePairGenerator": argmap.MaxOutDegreePreferencePairGenerator,
    "argmap.MaxArgsPreferencePairGenerator": argmap.MaxArgsPreferencePairGenerator,
    "argmap.MinLeafsPreferencePairGenerator": argmap.MinLeafsPreferencePairGenerator,
    "argmap.MaxAttacksPreferencePairGenerator": argmap.MaxAttacksPreferencePairGenerator,
    "argmap.MaxSupportsPreferencePairGenerator": argmap.MaxSupportsPreferencePairGenerator,
    "argmap.ShortClaimsPreferencePairGenerator": argmap.ShortClaimsPreferencePairGenerator,
    "argmap.LongClaimsPreferencePairGenerator": argmap.LongClaimsPreferencePairGenerator,
    "argmap.ShortLabelsPreferencePairGenerator": argmap.ShortLabelsPreferencePairGenerator,
    "argmap.DiverseLabelsPreferencePairGenerator": argmap.DiverseLabelsPreferencePairGenerator,
    "argmap.ArgumentClaimSizePreferencePairGenerator": argmap.ArgumentClaimSizePreferencePairGenerator,
    "argmap.IndependentWordingPreferencePairGenerator": argmap.IndependentWordingPreferencePairGenerator,
    "argmap.SourceTextProximityPreferencePairGenerator": argmap.SourceTextProximityPreferencePairGenerator,
    # module infreco
    "infreco.InformalReco": infreco.InformalReco,
    "infreco.InfRecoProblem": infreco.InfRecoProblem,
    "infreco.InfRecoProblemGenerator": infreco.InfRecoProblemGenerator,
    "infreco.InfRecoJudge": infreco.InfRecoJudge,
    "infreco.InfRecoFeedbackGenerator": infreco.InfRecoFeedbackGenerator,
    "infreco.NoUnusedPropsPreferencePairGenerator": infreco.NoUnusedPropsPreferencePairGenerator,
    "infreco.FewIntermediateConclusionsPreferencePairGenerator": infreco.FewIntermediateConclusionsPreferencePairGenerator,
    "infreco.ManyIntermediateConclusionsPreferencePairGenerator": infreco.ManyIntermediateConclusionsPreferencePairGenerator,
    "infreco.SimplicityPreferencePairGenerator": infreco.SimplicityPreferencePairGenerator,
    "infreco.VerbosityPreferencePairGenerator": infreco.VerbosityPreferencePairGenerator,
    "infreco.IndependentWordingPreferencePairGenerator": infreco.IndependentWordingPreferencePairGenerator,
    "infreco.SourceTextProximityPreferencePairGenerator": infreco.SourceTextProximityPreferencePairGenerator,
    # module logreco
    "logreco.LogicalReco": logreco.LogicalReco,
    "logreco.LogRecoProblem": logreco.LogRecoProblem,
    "logreco.LogRecoProblemGenerator": logreco.LogRecoProblemGenerator,
    "logreco.LogRecoJudge": logreco.LogRecoJudge,
    "logreco.LogRecoFeedbackGenerator": logreco.LogRecoFeedbackGenerator,
    "logreco.PredicateLogicPreferencePairGenerator": logreco.PredicateLogicPreferencePairGenerator,
    "logreco.FormalizationsFaithfulnessPreferencePairGenerator": logreco.FormalizationsFaithfulnessPreferencePairGenerator,
    "logreco.FewIntermediateConclusionsPreferencePairGenerator": logreco.FewIntermediateConclusionsPreferencePairGenerator,
    "logreco.ManyIntermediateConclusionsPreferencePairGenerator": logreco.ManyIntermediateConclusionsPreferencePairGenerator,
    "logreco.SourceTextProximityPreferencePairGenerator": logreco.SourceTextProximityPreferencePairGenerator,
    "logreco.IndependentWordingPreferencePairGenerator": logreco.IndependentWordingPreferencePairGenerator,
    "logreco.SimplicityPreferencePairGenerator": logreco.SimplicityPreferencePairGenerator,
    "logreco.VerbosityPreferencePairGenerator": logreco.VerbosityPreferencePairGenerator,
    # arganno_from_argmap
    "arganno_from_argmap.ArgannoFromArgmapProblem": arganno_from_argmap.ArgannoFromArgmapProblem,
    "arganno_from_argmap.ArgannoFromArgmapProblemGenerator": arganno_from_argmap.ArgannoFromArgmapProblemGenerator,
    "arganno_from_argmap.ArgmapTextProximityPreferencePairGenerator": arganno_from_argmap.ArgmapTextProximityPreferencePairGenerator,
    "arganno_from_argmap.ArgmapGraphProximityPreferencePairGenerator": arganno_from_argmap.ArgmapGraphProximityPreferencePairGenerator,
    # argmap_from_arganno
    "argmap_from_arganno.ArgmapFromArgannoProblem": argmap_from_arganno.ArgmapFromArgannoProblem,
    "argmap_from_arganno.ArgmapFromArgannoProblemGenerator": argmap_from_arganno.ArgmapFromArgannoProblemGenerator,
    "argmap_from_arganno.AnnotationTextProximityPreferencePairGenerator": argmap_from_arganno.AnnotationTextProximityPreferencePairGenerator,
    "argmap_from_arganno.AnnotationGraphProximityPreferencePairGenerator": argmap_from_arganno.AnnotationGraphProximityPreferencePairGenerator,
    # infreco_from_arganno
    "infreco_from_arganno.InfRecoFromArgAnnoProblem": infreco_from_arganno.InfRecoFromArgAnnoProblem,
    "infreco_from_arganno.InfRecoFromArgAnnoProblemGenerator": infreco_from_arganno.InfRecoFromArgAnnoProblemGenerator,
    "infreco_from_arganno.AnnotationProximityPreferencePairGenerator": infreco_from_arganno.AnnotationProximityPreferencePairGenerator,
    # logreco_from_infreco
    "logreco_from_infreco.LogrecoFromInfrecoProblem": logreco_from_infreco.LogrecoFromInfrecoProblem,
    "logreco_from_infreco.LogrecoFromInfrecoProblemGenerator": logreco_from_infreco.LogrecoFromInfrecoProblemGenerator,
    "logreco_from_infreco.InfrecoProximityPreferencePairGenerator": logreco_from_infreco.InfrecoProximityPreferencePairGenerator,
    # arganno_plus_infreco
    "arganno_plus_infreco.ArgannoPlusInfreco": arganno_plus_infreco.ArgannoPlusInfreco,
    "arganno_plus_infreco.ArgannoPlusInfrecoProblem": arganno_plus_infreco.ArgannoPlusInfrecoProblem,
    "arganno_plus_infreco.ArgannoPlusInfrecoProblemGenerator": arganno_plus_infreco.ArgannoPlusInfrecoProblemGenerator,
    "arganno_plus_infreco.ArgannoPlusInfrecoJudge": arganno_plus_infreco.ArgannoPlusInfrecoJudge,
    "arganno_plus_infreco.AnnotationProximityPreferencePairGenerator": arganno_plus_infreco.AnnotationProximityPreferencePairGenerator,
    # arganno_plus_logreco
    "arganno_plus_logreco.ArgannoPlusLogReco": arganno_plus_logreco.ArgannoPlusLogReco,
    "arganno_plus_logreco.ArgannoPlusLogRecoProblem": arganno_plus_logreco.ArgannoPlusLogRecoProblem,
    "arganno_plus_logreco.ArgannoPlusLogRecoProblemGenerator": arganno_plus_logreco.ArgannoPlusLogRecoProblemGenerator,
    "arganno_plus_logreco.ArgannoPlusLogRecoJudge": arganno_plus_logreco.ArgannoPlusLogRecoJudge,
    # argmap_plus_arganno
    "argmap_plus_arganno.ArgmapPlusArganno": argmap_plus_arganno.ArgmapPlusArganno,
    "argmap_plus_arganno.ArgmapPlusArgannoProblem": argmap_plus_arganno.ArgmapPlusArgannoProblem,
    "argmap_plus_arganno.ArgmapPlusArgannoProblemGenerator": argmap_plus_arganno.ArgmapPlusArgannoProblemGenerator,
    "argmap_plus_arganno.ArgmapPlusArgannoJudge": argmap_plus_arganno.ArgmapPlusArgannoJudge,
    "argmap_plus_arganno.AnnotationProximityPreferencePairGenerator": argmap_plus_arganno.AnnotationProximityPreferencePairGenerator,
    # argmap_plus_infreco
    "argmap_plus_infreco.ArgmapPlusInfreco": argmap_plus_infreco.ArgmapPlusInfreco,
    "argmap_plus_infreco.ArgmapPlusInfrecoProblem": argmap_plus_infreco.ArgmapPlusInfrecoProblem,
    "argmap_plus_infreco.ArgmapPlusInfrecoProblemGenerator": argmap_plus_infreco.ArgmapPlusInfrecoProblemGenerator,
    "argmap_plus_infreco.ArgmapPlusInfrecoJudge": argmap_plus_infreco.ArgmapPlusInfrecoJudge,
    "argmap_plus_infreco.SimplicityPreferencePairGenerator": argmap_plus_infreco.SimplicityPreferencePairGenerator,
    "argmap_plus_infreco.ConnectednessPreferencePairGeneratorCT": argmap_plus_infreco.ConnectednessPreferencePairGeneratorCT,
    "argmap_plus_infreco.MaxArgsPreferencePairGeneratorCT": argmap_plus_infreco.MaxArgsPreferencePairGeneratorCT,
    "argmap_plus_infreco.MaxSupportsPreferencePairGeneratorCT": argmap_plus_infreco.MaxSupportsPreferencePairGeneratorCT,
    "argmap_plus_infreco.MaxAttacksPreferencePairGeneratorCT": argmap_plus_infreco.MaxAttacksPreferencePairGeneratorCT,
    "argmap_plus_infreco.SourceTextProximityPreferencePairGeneratorCT": argmap_plus_infreco.SourceTextProximityPreferencePairGeneratorCT,
    # argmap_plus_logreco
    "argmap_plus_logreco.ArgmapPlusLogreco": argmap_plus_logreco.ArgmapPlusLogreco,
    "argmap_plus_logreco.ArgmapPlusLogrecoProblem": argmap_plus_logreco.ArgmapPlusLogrecoProblem,
    "argmap_plus_logreco.ArgmapPlusLogrecoProblemGenerator": argmap_plus_logreco.ArgmapPlusLogrecoProblemGenerator,
    "argmap_plus_logreco.ArgmapPlusLogrecoJudge": argmap_plus_logreco.ArgmapPlusLogrecoJudge,
    "argmap_plus_logreco.GlobalFormalizationsFaithfulnessPreferencePairGenerator": argmap_plus_logreco.GlobalFormalizationsFaithfulnessPreferencePairGenerator,
    # argmap_plus_arganno_plus_logreco
    "argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogreco": argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogreco,
    "argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoProblem": argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoProblem,
    "argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoProblemGenerator": argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoProblemGenerator,
    "argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoJudge": argmap_plus_arganno_plus_logreco.ArgmapPlusArgannoPlusLogrecoJudge,
}


def get_class(class_name: str):
    """
    Get the class from the registry by its name.
    """
    if class_name not in _CLASS_REGISTRY:
        raise ValueError(f"Class {class_name} not found in registry.")
    return _CLASS_REGISTRY[class_name]