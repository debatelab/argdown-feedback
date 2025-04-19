from textwrap import dedent

from argdown_feedback.tasks.base import GenericSolutionGenerator



def test_postprocessor_onelinereps_1():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck:          
                  
        ```argdown
        A
          +> B
          +> B
          +> B
          +> B
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    lines = text.splitlines()
    expected = "\n".join(lines[:-2]+lines[-1:])
    assert postprocessed == expected


def test_postprocessor_onelinereps_2():
    sg = GenericSolutionGenerator()

    text = dedent("""
        This is a test input for the solution generator.
                  
        I get stuck, but text has less than 16 lines:
                  
        ```argdown
        A
          +> B
          +> B
          +> B
          +> B
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    expected = text
    assert postprocessed == expected


def test_postprocessor_onelinereps_3():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I do not get stuck, cause I finish with a new line:         
                  
        ```argdown
        A
          +> B
          +> B
          +> B
          +> B
        C
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    expected = text
    assert postprocessed == expected


def test_postprocessor_onelinereps_4():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck, but not enough repetitions to remove:         
                  
        ```argdown
        A
          +> B
          +> B
          +> B
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    expected = text
    assert postprocessed == expected


def test_postprocessor_twolinereps_1():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck:          
                  
        ```argdown
        A
          +> B
            +> C
          +> B
            +> C
          +> B
            +> C
          +> B
            +> C
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    lines = text.splitlines()
    expected = "\n".join(lines[:-3]+lines[-1:])
    print(expected)
    assert postprocessed == expected


def test_postprocessor_twolinereps_2():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck:          
                  
        ```argdown
        A
          +> B
            +> C
          +> B
            +> C
          +> B
            +> C
          +> B
            +> C
          +> B
            +> C
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    lines = text.splitlines()
    expected = "\n".join(lines[:-5]+lines[-1:])
    print(expected)
    assert postprocessed == expected


def test_postprocessor_threelinereps_1():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck:          
                  
        ```argdown
        A
          +> B
            +> C
            +> C
          +> B
            +> C
            +> C
          +> B
            +> C
            +> C
          +> B
            +> C
            +> C
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    lines = text.splitlines()
    expected = "\n".join(lines[:-4]+lines[-1:])
    print(expected)
    assert postprocessed == expected



def test_postprocessor_fourlinereps_1():
    sg = GenericSolutionGenerator()

    text = "\n".join(["some line"]*16)
    text += dedent("""
        This is a test input for the solution generator.
                  
        I get stuck:          
                  
        ```argdown
        A
          +> B
            +> C
            +> C
            +> D
          +> B
            +> C
            +> C
            +> D
          +> B
            +> C
            +> C
            +> D
          +> B
            +> C
            +> C
            +> D
          +>
    """).strip("\n ")

    postprocessed = sg.remove_repetitions(text)
    print(postprocessed)
    expected = text
    assert postprocessed == expected
