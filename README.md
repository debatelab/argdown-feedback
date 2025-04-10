# argdown-feedback

Argdown Verifiers & Data Generators for Hindsight Instruction Relabeling Preferences


### Installation

```
!pip install "git+https://github.com/debatelab/argdown-feedback"
```

### Usage

```python
import pprint

from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import InfRecoCompositeHandler
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.tasks.base import Evaluation
from argdown_feedback.verifiers.verification_request import VerificationRequest


handler = CompositeHandler(
    handlers=[
        DefaultProcessingHandler(),
        InfRecoCompositeHandler(),
    ]
)
request = VerificationRequest(inputs=snippet)
result = handler.handle(request)
evaluation = Evaluation.from_verification_request(result)

pprint.pprint(evaluation)
# Evaluation(is_valid=False,
#            artifacts={'all_declarations': None,
#                       'all_expressions': None,
#                       'argdown': <pyargdown.model.ArgdownMultiDiGraph object at 0x7c27e54ced50>,
#                       'argdown_map': None,
#                       'argdown_reco': None,
#                       'soup': None},
#            metrics={'01_HasArgumentsHandler': None,
#                     '02_HasUniqueArgumentHandler': None,
#                     '03_HasPCSHandler': None,
#                     '04_StartsWithPremiseHandler': None,
#                     '05_EndsWithConclusionHandler': None,
#                     '06_NotMultipleGistsHandler': None,
#                     '07_NoDuplicatePCSLabelsHandler': None,
#                     '08_HasLabelHandler': None,
#                     '09_HasGistHandler': 'The following arguments lack gists: '
#                                          '<Argument>',
#                     '10_HasInferenceDataHandler': None,
#                     '11_PropRefsExistHandler': None,
#                     '12_UsesAllPropsHandler': 'In <Argument>: Some '
#                                               'propositions are not explicitly '
#                                               'used in any inferences: (1).',
#                     '13_NoExtraPropositionsHandler': None,
#                     '14_OnlyGroundedDialecticalRelationsHandler': None,
#                     '15_NoPropInlineDataHandler': None,
#                     '16_NoArgInlineDataHandler': None})
```