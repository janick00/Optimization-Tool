"""Tests for the central topic configuration."""

import unittest

from gui.topic_registry import TOPIC_REGISTRY


class TopicRegistryTests(unittest.TestCase):
    def test_topic_identifiers_are_unique(self) -> None:
        identifiers = [topic.identifier for topic in TOPIC_REGISTRY]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_every_topic_has_a_label_and_page_factory(self) -> None:
        self.assertGreater(len(TOPIC_REGISTRY), 0)
        for topic in TOPIC_REGISTRY:
            self.assertTrue(topic.identifier)
            self.assertTrue(topic.label)
            self.assertTrue(callable(topic.page_factory))

