"""
Utility functions for report generation.
"""

import os
from typing import Dict, List, Any, TextIO
import numpy as np
from datetime import datetime
from scipy import stats
from typing import Tuple, Optional, Union

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

def safe_statistical_test(test_type: str, 
                         *args,
                         min_samples: int = 3,
                         **kwargs) -> Tuple[Optional[float], Optional[float], str]:
    """
    Safely perform statistical tests with sample size checks.
    
    Args:
        test_type: Type of test ('ttest_ind', 'ttest_1samp', 'pearsonr', 'f_oneway', etc.)
        *args: Arguments for the test
        min_samples: Minimum required samples per group
        **kwargs: Additional keyword arguments for the test
    
    Returns:
        Tuple containing:
        - statistic (or None if test fails)
        - p-value (or None if test fails)
        - message explaining the result or why the test failed
    """
    try:
        if test_type == 'ttest_ind':
            group1, group2 = args
            if len(group1) < min_samples or len(group2) < min_samples:
                return None, None, f"Insufficient sample size (minimum {min_samples} required per group)"
            stat, p = stats.ttest_ind(group1, group2, **kwargs)
            return stat, p, "Test completed successfully"
            
        elif test_type == 'ttest_1samp':
            data, popmean = args
            if len(data) < min_samples:
                return None, None, f"Insufficient sample size (minimum {min_samples} required)"
            stat, p = stats.ttest_1samp(data, popmean, **kwargs)
            return stat, p, "Test completed successfully"
            
        elif test_type == 'pearsonr':
            x, y = args
            if len(x) < min_samples or len(y) < min_samples:
                return None, None, f"Insufficient sample size (minimum {min_samples} required)"
            stat, p = stats.pearsonr(x, y)
            return stat, p, "Test completed successfully"
            
        elif test_type == 'f_oneway':
            groups = args
            if any(len(group) < min_samples for group in groups):
                return None, None, f"Insufficient sample size (minimum {min_samples} required per group)"
            stat, p = stats.f_oneway(*groups)
            return stat, p, "Test completed successfully"
            
        elif test_type == 'shapiro':
            data = args[0]
            if len(data) < min_samples:
                return None, None, f"Insufficient sample size (minimum {min_samples} required)"
            stat, p = stats.shapiro(data)
            return stat, p, "Test completed successfully"
            
        elif test_type == 'linregress':
            x, y = args
            if len(x) < min_samples or len(y) < min_samples:
                return None, None, f"Insufficient sample size (minimum {min_samples} required)"
            result = stats.linregress(x, y)
            return result.statistic, result.pvalue, "Test completed successfully"
            
        else:
            return None, None, f"Unsupported test type: {test_type}"
            
    except Exception as e:
        return None, None, f"Error performing {test_type}: {str(e)}"

def format_statistical_result(stat: Optional[float], 
                            p: Optional[float], 
                            message: str,
                            test_name: str = "Test") -> str:
    """
    Format statistical test results into a readable string.
    
    Args:
        stat: Test statistic
        p: P-value
        message: Message from the test
        test_name: Name of the test for the output
    
    Returns:
        Formatted string with test results
    """
    if stat is None or p is None:
        return f"{test_name}: {message}"
    else:
        return f"{test_name}: statistic = {stat:.3f}, p = {p:.3f}" 