import sys
import unittest
from pathlib import Path
from typing import List

from parameterized import parameterized

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.statements.statement_generation import (
    get_simple_agents,
    DummyGenerator,
    NamedChatbotPersonalizationGenerator,
    LLMGenerator,
)

agents = get_simple_agents()

test_cases = [
    (
        agents[:10],
        2,
    ),
    (            
        agents[10:18],
        3,
    ),
    (            
        agents[30:48],
        5,
    ),
]

class TestDummyGenerator(unittest.TestCase):
    @parameterized.expand(test_cases)
    def test_generate(self, agents: List[Agent], num_statements: int):
        for statement_length in [2, 5, 10, 17]:
            generator = DummyGenerator(num_statements=num_statements, statement_length=statement_length)

            statements, logs = generator.generate(agents=agents)

            assert len(statements)==num_statements
            for statement in statements:
                assert len(statement)==statement_length

    @parameterized.expand(test_cases)
    def test_generate_with_context(self, agents: List[Agent], num_statements: int):
        for statement_length in [2, 5, 10, 17]:
            generator = DummyGenerator(num_statements=num_statements, statement_length=statement_length)

            results, logs = generator.generate_with_context(agents=agents)
            assert len(results)==num_statements
            for result in results:
                assert len(result.statement)==statement_length
                assert result.generation_method==generator.name


class TestLLMGenerator(unittest.TestCase):
    @parameterized.expand(test_cases[:2])
    def test_generate(self, agents: List[Agent], num_statements: int):
        generator = LLMGenerator(num_statements=num_statements, model="gpt-4o-mini")

        statements, logs = generator.generate(agents=agents)

        assert len(statements)==num_statements
        for statement in statements:
            assert len(statement.strip())>4

        assert len(logs)==1  # The method generates statements in a single call

    @parameterized.expand(test_cases[:2])
    def test_generate_with_context(self, agents: List[Agent], num_statements: int):
        generator = LLMGenerator(num_statements=num_statements, model="gpt-4o-mini")

        results, logs = generator.generate_with_context(agents=agents)
        assert len(results)==num_statements
        for result in results:
            assert len(result.statement.strip())>4
            assert result.generation_method==generator.name

        assert len(logs)==1  # The method generates statements in a single call


class TestNamedChatbotPersonalizationGenerator(unittest.TestCase):
    @parameterized.expand([
        (agents[:5],),
    ])
    def test_generate(self, agents: List[Agent]):
        generator = NamedChatbotPersonalizationGenerator(model="gpt-4o-mini")

        statements, logs = generator.generate(agents=agents)

        assert len(statements)==1
        for statement in statements:
            assert len(statement.strip())>4

        assert len(logs)==1

    @parameterized.expand([
        (agents[:5],),
    ])
    def test_generate_with_context(self, agents: List[Agent]):
        generator = NamedChatbotPersonalizationGenerator(model="gpt-4o-mini")

        results, logs = generator.generate_with_context(agents=agents)
        assert len(results)==1
        for result in results:
            assert len(result.statement.strip())>4
            assert result.generation_method==generator.name

        assert len(logs)==1 