from rest_framework import serializers
from .models import Tests

class TestsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tests
        fields = ['id', 'description', 'result', 'device_id', 'test_type', 'created_at']