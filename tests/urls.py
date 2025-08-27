from django.urls import path
from . import views

urlpatterns = [
    path('analyze-test/', views.analyze_test, name='analyze_test'),
    path('test-history/', views.ListHistoryView.as_view(), name='test_history')
]
