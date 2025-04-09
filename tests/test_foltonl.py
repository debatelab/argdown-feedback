from nltk.sem.logic import Expression  # type: ignore
from argdown_hirpo.logic.fol_to_nl import FOL2NLTranslator

class TestFOL2NLTranslator:
    
    def test_get_placeholders(self):
        scheme = "This is a {test} with {multiple} placeholders"
        placeholders = FOL2NLTranslator._get_placeholders(scheme)
        assert sorted(placeholders) == sorted(["test", "multiple"])
        
        scheme = "No placeholders here"
        placeholders = FOL2NLTranslator._get_placeholders(scheme)
        assert placeholders == []
        
        scheme = "{duplicate} and {duplicate}"
        placeholders = FOL2NLTranslator._get_placeholders(scheme)
        assert placeholders == ["duplicate"]
    
    def test_translate_unary_predicate(self):
        # Test unary predicate: P(x)
        expr = Expression.fromstring('P(x)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is {P}"
        # Test unary predicate: Pet(x)
        expr = Expression.fromstring('Pet(x)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is {Pet}"
        # Test unary predicate: P(pete)
        expr = Expression.fromstring('P(pete)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{pete} is {P}"

    def test_translate_binary_relation(self):
        # Test binary relation: R(x, y)
        expr = Expression.fromstring('R(x, y)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} stands in relation {R} to {y}"
        # Test binary relation: R(x, y)
        expr = Expression.fromstring('Like(pete, paul)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{pete} stands in relation {Like} to {paul}"
        
    def test_translate_negation(self):
        # Test negation: ~P(x)
        expr = Expression.fromstring('-P(x)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is not {P}"
        
        # Test negation of complex expression: ~(P(x) & Q(y))
        expr = Expression.fromstring('-(P(x) & Q(y))')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "it is false that {x} is {P} and {y} is {Q}"
        
    def test_translate_conjunction(self):
        # Test conjunction: P(x) & Q(y)
        expr = Expression.fromstring('P(x) & Q(y)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is {P} and {y} is {Q}"
        
    def test_translate_disjunction(self):
        # Test disjunction: P(x) | Q(y)
        expr = Expression.fromstring('P(x) | Q(y)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is {P} or {y} is {Q}"
        
    def test_translate_implication(self):
        # Test implication: P(x) -> Q(y)
        expr = Expression.fromstring('P(x) -> Q(y)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "if {x} is {P}, then {y} is {Q}"
        
    def test_translate_biconditional(self):
        # Test biconditional: P(x) <-> Q(y)
        expr = Expression.fromstring('P(x) <-> Q(y)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "if and only if {x} is {P}, then {y} is {Q}"
        
    def test_translate_universal_quantification(self):
        # Test universal quantification: all x.P(x)
        expr = Expression.fromstring('all x.P(x)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "for every {x} it holds that {x} is {P}"
        
    def test_translate_existential_quantification(self):
        # Test existential quantification: exists x.P(x)
        expr = Expression.fromstring('exists x.P(x)')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "there exists a {x} such that {x} is {P}"
        
    def test_translate_equality(self):
        # Test equality: x = y
        expr = Expression.fromstring('x = y')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{x} is identical with {y}"
        # Test equality: pete = peter
        expr = Expression.fromstring('pete = peter')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{pete} is identical with {peter}"
        
    def test_translate_complex_expression(self):
        # Test complex expression: all x.(P(x) -> exists y.(R(x,y) & Q(y)))
        expr = Expression.fromstring('all x.(P(x) -> exists y.(R(x,y) & Q(y)))')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        expected = "for every {x} it holds that if {x} is {P}, then there exists a {y} such that {x} stands in relation {R} to {y} and {y} is {Q}"
        assert result == expected
        
    def test_translate_to_nl_sentence(self):
        # Test translation with declarations
        expr = Expression.fromstring('P(x)')
        declarations = {"P": "happy", "x": "John"}
        result = FOL2NLTranslator.translate_to_nl_sentence(expr, declarations)
        assert result == "John is happy"
        
        expr = Expression.fromstring('all x.(H(x) -> M(x))')
        declarations = {"H": "human", "M": "mortal", "x": "being"}
        result = FOL2NLTranslator.translate_to_nl_sentence(expr, declarations)
        assert result == "for every being it holds that if being is human, then being is mortal"

    def test_translate_propositional_logic(self):
        # Test propositional logic: P & Q
        expr = Expression.fromstring('P & Q')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{P} and {Q}"
        
        # Test propositional logic: P & Q
        expr = Expression.fromstring('PP & QQ')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{PP} and {QQ}"

        # Test propositional logic: P | Q
        expr = Expression.fromstring('P | Q')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "{P} or {Q}"
        
        # Test propositional logic: P -> Q
        expr = Expression.fromstring('P -> Q')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "if {P}, then {Q}"
        
        # Test propositional logic: P <-> Q
        expr = Expression.fromstring('P <-> Q')
        result = FOL2NLTranslator.translate_to_nl_scheme(expr)
        assert result == "if and only if {P}, then {Q}"