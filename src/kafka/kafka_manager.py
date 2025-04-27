#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/kafka/kafka_manager.py - Kafka integration manager
Author: YourName
Date: 2025-04-27
Description: Manages Kafka connections and message processing
"""

from confluent_kafka import Consumer, Producer
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class KafkaManager:
    """Manages Kafka connections and message handling"""

    def __init__(self, config=None):
        """Initialize Kafka manager

        Args:
            config (dict): Kafka configuration
        """
        # Default Kafka configuration
        self.config = {
            'bootstrap.servers': 'pkc-312o0.ap-southeast-1.aws.confluent.cloud:9092',
            'group.id': 'realtime-recommender',
            'auto.offset.reset': 'earliest',
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': '***',  # Replace with environment variable or secure storage
            'sasl.password': '***'  # Replace with environment variable or secure storage
        }

        # Update with provided config
        if config:
            self.config.update(config)

        self.consumer = None
        self.producer = None

    def create_consumer(self, topics):
        """Create Kafka consumer

        Args:
            topics (list): List of topics to subscribe to
        """
        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)
        return self.consumer

    def create_producer(self):
        """Create Kafka producer"""
        self.producer = Producer(self.config)
        return self.producer

    def process_message(self, msg):
        """Process Kafka message

        Args:
            msg: Kafka message

        Returns:
            dict: Parsed message data
        """
        try:
            value = msg.value().decode('utf-8')
            data = json.loads(value)
            return data
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return None

    def send_message(self, topic, data):
        """Send message to Kafka topic

        Args:
            topic (str): Topic name
            data (dict): Message data to send
        """
        if not self.producer:
            self.create_producer()

        try:
            self.producer.produce(topic, json.dumps(data).encode('utf-8'))
            self.producer.flush()
            return True
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return False

    def close(self):
        """Close Kafka connections"""
        if self.consumer:
            self.consumer.close()
        # Producer doesn't need explicit closing