"""
Utility functions for report generation.
"""

import os
from typing import Dict, List, Any, TextIO
import numpy as np
from datetime import datetime

def create_report_header(
    title: str,
    analysis_type: str,
    file: TextIO
) -> None:
    """
    Create a standardized header for reports.
    
    Args:
        title: Report title
        analysis_type: Type of analysis
        file: File object to write to
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file.write("=" * 80 + "\n")
    file.write(f"{title}\n")
    file.write("=" * 80 + "\n")
    file.write(f"Analysis Type: {analysis_type}\n")
    file.write(f"Generated on: {current_time}\n")
    file.write("-" * 80 + "\n\n")

def format_statistical_result(
    test_name: str,
    statistic: float,
    pvalue: float,
    additional_info: Dict[str, Any] = None
) -> str:
    """
    Format statistical test results.
    
    Args:
        test_name: Name of the statistical test
        statistic: Test statistic value
        pvalue: P-value from the test
        additional_info: Additional information to include
    
    Returns:
        str: Formatted statistical result
    """
    result = f"{test_name}:\n"
    result += f"  Statistic: {statistic:.4f}\n"
    result += f"  P-value: {pvalue:.4f}"
    
    if additional_info:
        for key, value in additional_info.items():
            if isinstance(value, float):
                result += f"\n  {key}: {value:.4f}"
            else:
                result += f"\n  {key}: {value}"
    
    return result + "\n"

def create_section_header(
    section_title: str,
    file: TextIO
) -> None:
    """
    Create a standardized section header in reports.
    
    Args:
        section_title: Title of the section
        file: File object to write to
    """
    file.write("\n" + "-" * 40 + "\n")
    file.write(f"{section_title}\n")
    file.write("-" * 40 + "\n\n")

def format_descriptive_stats(
    data: np.ndarray,
    name: str
) -> str:
    """
    Format descriptive statistics for a variable.
    
    Args:
        data: Data array
        name: Variable name
    
    Returns:
        str: Formatted descriptive statistics
    """
    return (
        f"{name} Summary Statistics:\n"
        f"  Mean: {np.mean(data):.4f}\n"
        f"  Std: {np.std(data):.4f}\n"
        f"  Min: {np.min(data):.4f}\n"
        f"  Max: {np.max(data):.4f}\n"
        f"  Median: {np.median(data):.4f}\n"
    ) 