# tools/data_analysis_tool.py

"""
Tool for analyzing numerical or structured data.
"""
from typing import Dict, List, Any, Optional, Union
import json
import re
import pandas as pd
import numpy as np
import io
from pathlib import Path


class DataAnalysisTool:
    """Tool for analyzing structured or numerical data encountered during research."""
    
    def __init__(self):
        """Initialize the DataAnalysisTool."""
        self.supported_formats = ["csv", "json", "excel", "txt"]
        self.supported_analysis_types = [
            "descriptive_statistics", 
            "correlation", 
            "trend_analysis", 
            "comparison", 
            "frequency", 
            "distribution",
            "regression",
            "clustering",
            "time_series",
            "outlier_detection"
        ]
    
    def analyze(self, 
                data: Any, 
                analysis_type: str, 
                parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze data according to the specified analysis type.
        
        Args:
            data: The data to analyze (can be in various formats: string, dict, list, 
                  pandas DataFrame, or file path)
            analysis_type: The type of analysis to perform
            parameters: Optional parameters for the analysis
                
        Returns:
            A dictionary containing the analysis results
        """
        if parameters is None:
            parameters = {}
        
        # Convert data to pandas DataFrame
        df = self._convert_to_dataframe(data)
        
        # Choose analysis method based on analysis_type
        if analysis_type.lower() == "descriptive_statistics":
            result = self._descriptive_statistics(df, parameters)
        elif analysis_type.lower() == "correlation":
            result = self._correlation_analysis(df, parameters)
        elif analysis_type.lower() == "trend_analysis":
            result = self._trend_analysis(df, parameters)
        elif analysis_type.lower() == "comparison":
            result = self._comparison_analysis(df, parameters)
        elif analysis_type.lower() == "frequency":
            result = self._frequency_analysis(df, parameters)
        elif analysis_type.lower() == "distribution":
            result = self._distribution_analysis(df, parameters)
        elif analysis_type.lower() == "regression":
            result = self._regression_analysis(df, parameters)
        elif analysis_type.lower() == "clustering":
            result = self._clustering_analysis(df, parameters)
        elif analysis_type.lower() == "time_series":
            result = self._time_series_analysis(df, parameters)
        elif analysis_type.lower() == "outlier_detection":
            result = self._outlier_detection(df, parameters)
        else:
            # Default to basic analysis
            result = self._basic_analysis(df, parameters)
        
        return {
            "analysis_type": analysis_type,
            "parameters": parameters,
            "results": result
        }
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert input data to a pandas DataFrame.
        
        Handles various input formats:
        - pandas DataFrame (returned as-is)
        - string (parsed as CSV, JSON, or file path)
        - dict or list (converted to DataFrame)
        - numpy array (converted to DataFrame)
        """
        # If already a DataFrame, return as is
        if isinstance(data, pd.DataFrame):
            return data
        
        # If the data is a file path, read the file
        if isinstance(data, str) and (Path(data).exists() or data.startswith('http')):
            try:
                file_extension = Path(data).suffix.lower()[1:] if not data.startswith('http') else 'csv'
                
                if file_extension == 'csv':
                    return pd.read_csv(data)
                elif file_extension in ['xls', 'xlsx']:
                    return pd.read_excel(data)
                elif file_extension == 'json':
                    return pd.read_json(data)
                elif file_extension == 'txt':
                    # Try to auto-detect format for txt files
                    with open(data, 'r') as f:
                        content = f.read()
                        if content.strip().startswith('{') or content.strip().startswith('['):
                            return pd.read_json(data)
                        else:
                            return pd.read_csv(data, sep=None, engine='python')
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")
            except Exception as e:
                raise ValueError(f"Failed to read file: {e}")
        
        # If data is a string, try to parse it
        if isinstance(data, str):
            try:
                # Try to parse as JSON
                parsed_data = json.loads(data)
                return pd.DataFrame(parsed_data)
            except json.JSONDecodeError:
                # If not JSON, try to parse as CSV
                try:
                    return pd.read_csv(io.StringIO(data))
                except:
                    # If not CSV, try to parse as TSV or other delimiters
                    try:
                        return pd.read_csv(io.StringIO(data), sep=None, engine='python')
                    except:
                        raise ValueError("Unable to parse string data as JSON, CSV, or other delimited format")
        
        # If data is a dict, convert to DataFrame
        if isinstance(data, dict):
            return pd.DataFrame.from_dict(data, orient='index' if not isinstance(next(iter(data.values()), None), dict) else 'columns')
        
        # If data is a list, convert to DataFrame
        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return pd.DataFrame(data)
            elif all(isinstance(item, (int, float)) for item in data):
                return pd.DataFrame({'value': data})
            else:
                return pd.DataFrame(data)
        
        # If data is a numpy array, convert to DataFrame
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        
        # If we can't convert, raise error
        raise ValueError(f"Cannot convert data of type {type(data)} to DataFrame")
    
    def _descriptive_statistics(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate descriptive statistics for the data using pandas."""
        result = {}
        
        # Get columns to analyze (default: all numeric columns)
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {"error": "No numeric columns found for analysis"}
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
        
        # Filter columns that actually exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        for column in columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
                
            # Get column stats
            stats = df[column].describe().to_dict()
            
            # Add additional statistics not included in describe()
            stats['median'] = df[column].median()
            stats['mode'] = df[column].mode().iloc[0] if not df[column].mode().empty else None
            stats['variance'] = df[column].var()
            stats['skewness'] = df[column].skew()
            stats['kurtosis'] = df[column].kurtosis()
            stats['range'] = df[column].max() - df[column].min()
            stats['iqr'] = df[column].quantile(0.75) - df[column].quantile(0.25)
            stats['missing_values'] = df[column].isna().sum()
            stats['missing_percentage'] = (df[column].isna().sum() / len(df)) * 100
            
            result[column] = stats
            
        return result
    
    def _correlation_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlations between variables in the data using pandas."""
        result = {"correlations": {}}
        
        # Get method parameter (default: pearson)
        method = parameters.get('method', 'pearson').lower()
        if method not in ['pearson', 'kendall', 'spearman']:
            method = 'pearson'
        
        # Get columns to analyze (default: all numeric columns)
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {"error": "No numeric columns found for correlation analysis"}
            
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
            
        # Filter columns that actually exist in the dataframe
        columns = [col for col in columns if col in df.columns]
            
        # Filter to include only specified numeric columns
        numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Convert to dictionary format
        corr_dict = {}
        for col1 in corr_matrix.columns:
            corr_dict[col1] = {}
            for col2 in corr_matrix.columns:
                if col1 != col2:  # Skip self-correlations
                    corr_dict[col1][col2] = corr_matrix.loc[col1, col2]
        
        result["correlations"] = corr_dict
        
        # Find strongest positive and negative correlations
        strongest_positive = None
        strongest_negative = None
        max_positive = -1
        max_negative = 1
        
        for col1 in corr_dict:
            for col2 in corr_dict[col1]:
                corr_value = corr_dict[col1][col2]
                if corr_value > max_positive:
                    max_positive = corr_value
                    strongest_positive = (col1, col2)
                if corr_value < max_negative:
                    max_negative = corr_value
                    strongest_negative = (col1, col2)
        
        if strongest_positive:
            result["strongest_positive_correlation"] = {
                "variables": strongest_positive,
                "value": max_positive
            }
            
        if strongest_negative:
            result["strongest_negative_correlation"] = {
                "variables": strongest_negative,
                "value": max_negative
            }
        
        return result
    
    def _trend_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in time-series data."""
        result = {"trends": {}}
        
        # Extract time column and value column from parameters
        time_column = parameters.get("time_column")
        value_column = parameters.get("value_column")
        
        # If columns aren't specified, try to detect a date column
        if not time_column:
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 
                            ('date' in col.lower() or 'time' in col.lower())]
            if date_columns:
                time_column = date_columns[0]
            else:
                return {"error": "No time column specified or detected for trend analysis"}
        
        # If value column isn't specified, use the first numeric column that's not the time column
        if not value_column:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                value_column = numeric_columns[0]
            else:
                return {"error": "No value column specified or detected for trend analysis"}
        
        # Ensure time_column and value_column exist in the dataframe
        if time_column not in df.columns:
            return {"error": f"Time column '{time_column}' not found in dataframe"}
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in dataframe"}
        
        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except:
                # If conversion fails, use index as time
                result["warning"] = f"Could not convert '{time_column}' to datetime. Using row index instead."
                df['_index'] = df.index
                time_column = '_index'
        
        # Sort data by time
        sorted_df = df.sort_values(by=time_column)
        
        # Extract values
        times = sorted_df[time_column].tolist()
        values = sorted_df[value_column].tolist()
        
        # Basic trend metrics
        if len(values) >= 2:
            # Calculate overall change and percent change
            start_value = values[0]
            end_value = values[-1]
            change = end_value - start_value
            percent_change = (change / abs(start_value)) * 100 if start_value != 0 else float('inf' if end_value > 0 else 0)
            
            # Fit a linear trend line
            x = np.arange(len(values))
            y = np.array(values)
            slope, intercept = np.polyfit(x, y, 1)
            
            # Determine trend direction based on slope
            if slope > 0.05:  # Arbitrary threshold
                direction = "increasing"
            elif slope < -0.05:  # Arbitrary threshold
                direction = "decreasing"
            else:
                direction = "stable"
            
            # Calculate moving average
            window_size = min(parameters.get("window_size", 3), len(values))
            moving_avg = sorted_df[value_column].rolling(window=window_size).mean().tolist()
            
            # Calculate weighted moving average (more weight to recent values)
            weights = np.arange(1, window_size + 1)
            wma = sorted_df[value_column].rolling(window=window_size).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True).tolist()
            
            # Calculate rate of change
            roc = sorted_df[value_column].pct_change().tolist()
            
            # Store results
            result["trends"][value_column] = {
                "start_value": start_value,
                "end_value": end_value,
                "change": change,
                "percent_change": percent_change,
                "slope": slope,
                "intercept": intercept,
                "direction": direction,
                "moving_average": moving_avg[-5:],  # Last 5 values only to keep response size manageable
                "weighted_moving_average": wma[-5:],  # Last 5 values only
                "rate_of_change": roc[-5:],  # Last 5 values only
            }
            
            # Seasonality detection if we have enough data points
            if len(values) >= 12 and pd.api.types.is_datetime64_any_dtype(sorted_df[time_column]):
                # Check if data has monthly or quarterly patterns
                temp_df = sorted_df.copy()
                temp_df['month'] = temp_df[time_column].dt.month
                monthly_avg = temp_df.groupby('month')[value_column].mean().tolist()
                
                # Detect if there's significant variation by month (simplistic approach)
                monthly_variation = np.std(monthly_avg) / np.mean(monthly_avg) if np.mean(monthly_avg) != 0 else 0
                
                if monthly_variation > 0.1:  # Arbitrary threshold
                    result["trends"][value_column]["seasonality"] = {
                        "detected": True,
                        "monthly_pattern": {i+1: v for i, v in enumerate(monthly_avg)}
                    }
                else:
                    result["trends"][value_column]["seasonality"] = {"detected": False}
        
        return result
    
    def _comparison_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different groups or categories in the data."""
        result = {"comparisons": {}}
        
        # Extract group column and value column from parameters
        group_column = parameters.get("group_column")
        value_column = parameters.get("value_column")
        
        # If group column isn't specified, try to detect a categorical column
        if not group_column:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_columns:
                group_column = categorical_columns[0]
            else:
                return {"error": "No group column specified or detected for comparison analysis"}
        
        # If value column isn't specified, use the first numeric column
        if not value_column:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                value_column = numeric_columns[0]
            else:
                return {"error": "No value column specified or detected for comparison analysis"}
        
        # Ensure group_column and value_column exist in the dataframe
        if group_column not in df.columns:
            return {"error": f"Group column '{group_column}' not found in dataframe"}
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in dataframe"}
        
        # Group data and calculate statistics
        grouped = df.groupby(group_column)[value_column]
        
        # Calculate statistics for each group
        group_stats = {}
        for group_name, group_data in grouped:
            stats = group_data.describe().to_dict()
            
            # Add additional statistics
            stats['median'] = group_data.median()
            stats['variance'] = group_data.var()
            stats['skewness'] = group_data.skew()
            stats['kurtosis'] = group_data.kurtosis()
            stats['range'] = group_data.max() - group_data.min()
            stats['iqr'] = group_data.quantile(0.75) - group_data.quantile(0.25)
            
            group_stats[str(group_name)] = stats
        
        result["comparisons"] = group_stats
        
        # Find highest and lowest groups by mean
        if group_stats:
            group_means = {group: stats['mean'] for group, stats in group_stats.items()}
            max_group = max(group_means, key=group_means.get)
            min_group = min(group_means, key=group_means.get)
            
            result["group_comparison"] = {
                "highest_mean": {
                    "group": max_group,
                    "value": group_means[max_group]
                },
                "lowest_mean": {
                    "group": min_group,
                    "value": group_means[min_group]
                },
                "difference": group_means[max_group] - group_means[min_group],
                "percent_difference": ((group_means[max_group] - group_means[min_group]) / 
                                      abs(group_means[min_group])) * 100 if group_means[min_group] != 0 else float('inf')
            }
            
            # Add ANOVA test for statistical significance if scipy is available
            try:
                from scipy import stats as scipy_stats
                
                # Prepare data for ANOVA
                groups_data = [group_data.dropna().values for _, group_data in grouped]
                
                # Only perform ANOVA if we have at least 2 groups with data
                valid_groups = [g for g in groups_data if len(g) > 0]
                if len(valid_groups) >= 2:
                    f_val, p_val = scipy_stats.f_oneway(*valid_groups)
                    
                    result["group_comparison"]["anova_test"] = {
                        "f_value": f_val,
                        "p_value": p_val,
                        "significant_difference": p_val < 0.05
                    }
            except ImportError:
                pass  # Skip ANOVA if scipy is not available
        
        return result
    
    def _frequency_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the frequency distribution of categorical data."""
        result = {"frequencies": {}}
        
        # Extract category column from parameters
        category_column = parameters.get("category_column")
        
        # If category column isn't specified, try to detect a categorical column
        if not category_column:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_columns:
                category_column = categorical_columns[0]
            else:
                return {"error": "No category column specified or detected for frequency analysis"}
        
        # Ensure category_column exists in the dataframe
        if category_column not in df.columns:
            return {"error": f"Category column '{category_column}' not found in dataframe"}
        
        # Calculate value counts and percentages
        value_counts = df[category_column].value_counts()
        percentages = df[category_column].value_counts(normalize=True) * 100
        
        # Combine into a single result
        frequencies = {}
        for category in value_counts.index:
            frequencies[str(category)] = {
                "count": int(value_counts[category]),
                "percentage": float(percentages[category])
            }
        
        # Sort by count (descending)
        sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1]["count"], reverse=True))
        result["frequencies"] = sorted_frequencies
        
        # Find most and least common categories
        if sorted_frequencies:
            most_common = next(iter(sorted_frequencies.items()))
            least_common = next(reversed(sorted_frequencies.items()))
            
            result["most_common"] = {
                "category": most_common[0],
                "count": most_common[1]["count"],
                "percentage": most_common[1]["percentage"]
            }
            
            result["least_common"] = {
                "category": least_common[0],
                "count": least_common[1]["count"],
                "percentage": least_common[1]["percentage"]
            }
            
            # Calculate diversity metrics
            unique_count = len(sorted_frequencies)
            total_count = sum(item["count"] for item in sorted_frequencies.values())
            
            # Calculate entropy (Shannon diversity index)
            entropy = 0
            for item in sorted_frequencies.values():
                p = item["count"] / total_count
                entropy -= p * np.log(p) if p > 0 else 0
                
            result["diversity_metrics"] = {
                "unique_categories": unique_count,
                "entropy": entropy,
            }
        
        return result
    
    def _distribution_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the distribution of numeric data."""
        result = {"distributions": {}}
        
        # Get columns to analyze
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {"error": "No numeric columns found for distribution analysis"}
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
            
        # Filter columns that actually exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        for column in columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
                
            # Get column values (drop NA values)
            values = df[column].dropna().values
            
            if len(values) == 0:
                continue
                
            # Calculate basic distribution statistics
            dist_stats = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std_dev": float(np.std(values)),
                "skewness": float(df[column].skew()),
                "kurtosis": float(df[column].kurtosis())
            }
            
            # Calculate percentiles
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            dist_stats["percentiles"] = {
                f"p{p}": float(np.percentile(values, p)) for p in percentiles
            }
            
            # Determine if data is approximately normal
            try:
                from scipy import stats as scipy_stats
                _, p_value = scipy_stats.normaltest(values)
                dist_stats["normality_test"] = {
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05  # Common threshold for normality
                }
            except ImportError:
                # If scipy is not available, use a simple heuristic based on skewness and kurtosis
                is_normal = abs(dist_stats["skewness"]) < 0.5 and abs(dist_stats["kurtosis"]) < 1
                dist_stats["normality_test"] = {
                    "is_normal": is_normal,
                    "note": "Approximated using skewness and kurtosis thresholds"
                }
            
            # Create histograms (bin frequencies)
            try:
                hist, bin_edges = np.histogram(values, bins='auto')
                bins = []
                for i in range(len(hist)):
                    bins.append({
                        "range": [float(bin_edges[i]), float(bin_edges[i+1])],
                        "count": int(hist[i]),
                        "percentage": float(hist[i] / len(values) * 100)
                    })
                
                dist_stats["histogram"] = {
                    "bins": bins,
                    "bin_count": len(bins)
                }
            except:
                pass  # Skip histogram if it fails
            
            result["distributions"][column] = dist_stats
        
        return result
    
    def _regression_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regression analysis on the data."""
        result = {"regression": {}}
        
        # Get target and predictor columns
        target_column = parameters.get("target_column")
        predictor_columns = parameters.get("predictor_columns")
        
        # If target column isn't specified, try to detect a suitable column
        if not target_column:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                target_column = numeric_columns[-1]  # Use the last numeric column as target
            else:
                return {"error": "No target column specified or detected for regression analysis"}
        
        # Ensure target_column exists in the dataframe
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in dataframe"}
        
        # If predictor columns aren't specified, use all other numeric columns
        if not predictor_columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            predictor_columns = [col for col in numeric_columns if col != target_column]
            
            if not predictor_columns:
                return {"error": "No predictor columns specified or detected for regression analysis"}
        
        # Ensure predictor_columns is a list
        if isinstance(predictor_columns, str):
            predictor_columns = [predictor_columns]
            
        # Filter columns that actually exist in the dataframe
        predictor_columns = [col for col in predictor_columns if col in df.columns]
        
        # Skip non-numeric columns
        predictor_columns = [col for col in predictor_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if not predictor_columns:
            return {"error": "No valid numeric predictor columns found for regression analysis"}
            
        # Check for missing values
        complete_cases = df.dropna(subset=[target_column] + predictor_columns)
        if len(complete_cases) < len(df):
            result["warning"] = f"Dropped {len(df) - len(complete_cases)} rows with missing values"
        
        # Try to perform regression analysis
        try:
            # Import required libs
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Prepare data
            X = complete_cases[predictor_columns].values
            y = complete_cases[target_column].values
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Evaluate model
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Extract coefficients
            coefficients = {}
            for i, col in enumerate(predictor_columns):
                coefficients[col] = float(model.coef_[i])
            
            # Store results
            result["regression"] = {
                "model_type": "linear_regression",
                "target_column": target_column,
                "predictor_columns": predictor_columns,
                "intercept": float(model.intercept_),
                "coefficients": coefficients,
                "r_squared": float(r2),
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "sample_size": len(complete_cases)
            }
            
            # Calculate feature importance
            importance = {}
            for i, col in enumerate(predictor_columns):
                # Standardized coefficients (approximation)
                std_coef = model.coef_[i] * (np.std(X[:, i]) / np.std(y))
                importance[col] = abs(float(std_coef))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            result["regression"]["feature_importance"] = sorted_importance
            
        except ImportError:
            result["error"] = "scikit-learn is required for regression analysis but not installed"
        except Exception as e:
            result["error"] = f"Regression analysis failed: {str(e)}"
        
        return result
    
    def _clustering_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis on the data."""
        result = {"clustering": {}}
        
        # Get columns to use for clustering
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {"error": "No numeric columns found for clustering analysis"}
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
            
        # Filter columns that actually exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        # Skip non-numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if not columns:
            return {"error": "No valid numeric columns found for clustering analysis"}
            
        # Get clustering parameters
        n_clusters = parameters.get('n_clusters', 2)
        
        # Check for missing values
        complete_cases = df.dropna(subset=columns)
        if len(complete_cases) < len(df):
            result["warning"] = f"Dropped {len(df) - len(complete_cases)} rows with missing values"
            
        if len(complete_cases) == 0:
            return {"error": "No valid data points after dropping rows with missing values"}
            
        # Try to perform clustering analysis
        try:
            # Import required libs
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X = complete_cases[columns].values
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original dataframe
            temp_df = complete_cases.copy()
            temp_df['cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_df = temp_df[temp_df['cluster'] == cluster_id]
                
                # Calculate statistics for each numeric column
                col_stats = {}
                for col in columns:
                    col_stats[col] = {
                        "mean": float(cluster_df[col].mean()),
                        "median": float(cluster_df[col].median()),
                        "std_dev": float(cluster_df[col].std()),
                        "min": float(cluster_df[col].min()),
                        "max": float(cluster_df[col].max())
                    }
                
                cluster_stats[f"cluster_{cluster_id}"] = {
                    "size": int(len(cluster_df)),
                    "percentage": float(len(cluster_df) / len(temp_df) * 100),
                    "feature_statistics": col_stats
                }
            
            # Calculate cluster centers
            cluster_centers = {}
            for i, center in enumerate(kmeans.cluster_centers_):
                cluster_centers[f"cluster_{i}"] = {
                    columns[j]: float(center[j]) for j in range(len(columns))
                }
                
            # Calculate inertia (within-cluster sum of squares)
            inertia = float(kmeans.inertia_)
            
            # Store results
            result["clustering"] = {
                "method": "kmeans",
                "n_clusters": n_clusters,
                "cluster_sizes": {f"cluster_{i}": int((cluster_labels == i).sum()) for i in range(n_clusters)},
                "cluster_statistics": cluster_stats,
                "cluster_centers": cluster_centers,
                "inertia": inertia,
                "sample_size": len(complete_cases)
            }
            
        except ImportError:
            result["error"] = "scikit-learn is required for clustering analysis but not installed"
        except Exception as e:
            result["error"] = f"Clustering analysis failed: {str(e)}"
        
        return result
    
    def _time_series_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform time series analysis on the data."""
        result = {"time_series": {}}
        
        # Extract time column and value column from parameters
        time_column = parameters.get("time_column")
        value_column = parameters.get("value_column")
        
        # If columns aren't specified, try to detect a date column
        if not time_column:
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 
                            ('date' in col.lower() or 'time' in col.lower())]
            if date_columns:
                time_column = date_columns[0]
            else:
                return {"error": "No time column specified or detected for time series analysis"}
        
        # If value column isn't specified, use the first numeric column that's not the time column
        if not value_column:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                value_column = numeric_columns[0]
            else:
                return {"error": "No value column specified or detected for time series analysis"}
        
        # Ensure time_column and value_column exist in the dataframe
        if time_column not in df.columns:
            return {"error": f"Time column '{time_column}' not found in dataframe"}
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in dataframe"}
        
        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except:
                return {"error": f"Could not convert '{time_column}' to datetime"}
        
        # Sort data by time
        sorted_df = df.sort_values(by=time_column)
        
        # Set time column as index for time series operations
        ts_df = sorted_df.set_index(time_column)
        
        # Basic time series statistics
        ts_stats = {
            "start_date": str(ts_df.index.min()),
            "end_date": str(ts_df.index.max()),
            "duration": str(ts_df.index.max() - ts_df.index.min()),
            "data_points": len(ts_df),
            "mean": float(ts_df[value_column].mean()),
            "std_dev": float(ts_df[value_column].std()),
            "min": float(ts_df[value_column].min()),
            "max": float(ts_df[value_column].max())
        }
        
        # Check data frequency
        try:
            inferred_freq = pd.infer_freq(ts_df.index)
            ts_stats["frequency"] = str(inferred_freq) if inferred_freq else "irregular"
        except:
            ts_stats["frequency"] = "irregular"
        
        # Calculate trends
        if len(ts_df) >= 2:
            # Linear trend
            x = np.arange(len(ts_df))
            y = ts_df[value_column].values
            slope, intercept = np.polyfit(x, y, 1)
            
            ts_stats["trend"] = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
            
            # Calculate growth rate
            start_value = ts_df[value_column].iloc[0]
            end_value = ts_df[value_column].iloc[-1]
            time_diff = (ts_df.index[-1] - ts_df.index[0]).total_seconds() / (60*60*24*365.25)  # in years
            
            if time_diff > 0 and start_value != 0:
                annual_growth_rate = ((end_value / start_value) ** (1/time_diff) - 1) * 100
                ts_stats["annual_growth_rate"] = float(annual_growth_rate)
            
            # Moving averages
            window_size = min(parameters.get("window_size", 3), len(ts_df))
            
            # Simple moving average
            sma = ts_df[value_column].rolling(window=window_size).mean()
            
            # Exponential moving average
            ema = ts_df[value_column].ewm(span=window_size).mean()
            
            # Store last few values to keep response size manageable
            ts_stats["moving_averages"] = {
                "simple": [float(x) for x in sma.dropna().tail(5).tolist()],
                "exponential": [float(x) for x in ema.dropna().tail(5).tolist()]
            }
            
            # Check for seasonality if we have enough data
            if len(ts_df) >= 12:
                try:
                    # Add date components
                    temp_df = ts_df.copy()
                    temp_df['month'] = temp_df.index.month
                    temp_df['quarter'] = temp_df.index.quarter
                    temp_df['year'] = temp_df.index.year
                    
                    # Monthly averages
                    monthly_avg = temp_df.groupby('month')[value_column].mean()
                    
                    # Quarterly averages
                    quarterly_avg = temp_df.groupby('quarter')[value_column].mean()
                    
                    # Check for seasonal patterns
                    monthly_variation = np.std(monthly_avg) / np.mean(monthly_avg) if np.mean(monthly_avg) != 0 else 0
                    
                    ts_stats["seasonality"] = {
                        "detected": monthly_variation > 0.1,  # Arbitrary threshold
                        "monthly_pattern": {str(i): float(v) for i, v in enumerate(monthly_avg, 1)},
                        "quarterly_pattern": {str(i): float(v) for i, v in enumerate(quarterly_avg, 1)}
                    }
                except:
                    pass
        
        result["time_series"] = ts_stats
        
        return result
    
    def _outlier_detection(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in the data."""
        result = {"outliers": {}}
        
        # Get columns to analyze
        columns = parameters.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        if not columns:
            return {"error": "No numeric columns found for outlier detection"}
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
            
        # Filter columns that actually exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        # Get method for outlier detection
        method = parameters.get('method', 'iqr').lower()
        
        # Detect outliers for each column
        for column in columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
                
            # Get column values (drop NA values)
            values = df[column].dropna().values
            
            if len(values) == 0:
                continue
                
            outliers = []
            
            # IQR method (default)
            if method == 'iqr':
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, val in enumerate(values):
                    if val < lower_bound or val > upper_bound:
                        outliers.append({
                            "index": i,
                            "value": float(val),
                            "bound": "lower" if val < lower_bound else "upper"
                        })
                        
                result["outliers"][column] = {
                    "method": "IQR",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(values) * 100)
                }
                
            # Z-score method
            elif method == 'zscore':
                mean = np.mean(values)
                std = np.std(values)
                threshold = parameters.get('threshold', 3)  # Default is 3 standard deviations
                
                for i, val in enumerate(values):
                    z_score = (val - mean) / std if std > 0 else 0
                    if abs(z_score) > threshold:
                        outliers.append({
                            "index": i,
                            "value": float(val),
                            "z_score": float(z_score)
                        })
                        
                result["outliers"][column] = {
                    "method": "Z-score",
                    "mean": float(mean),
                    "std_dev": float(std),
                    "threshold": float(threshold),
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(values) * 100)
                }
            
            # Add sample of outliers (limited to 5 to keep response size manageable)
            if outliers:
                result["outliers"][column]["sample_outliers"] = outliers[:5]
        
        return result
    
    def _basic_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic analysis on the dataframe."""
        result = {}
        
        # Basic dataframe info
        result["shape"] = {
            "rows": df.shape[0],
            "columns": df.shape[1]
        }
        
        # Column information
        result["columns"] = df.columns.tolist()
        
        # Column types
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = "datetime"
            elif pd.api.types.is_categorical_dtype(df[col]):
                column_types[col] = "categorical"
            else:
                column_types[col] = "text"
                
        result["column_types"] = column_types
        
        # Missing values
        missing_values = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(df) * 100)
                }
                
        result["missing_values"] = missing_values if missing_values else {"info": "No missing values"}
        
        # Summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            numeric_stats = df[numeric_columns].describe().to_dict()
            
            # Convert numpy types to Python native types for better JSON serialization
            for col, stats in numeric_stats.items():
                for stat, value in stats.items():
                    numeric_stats[col][stat] = float(value)
                    
            result["numeric_summary"] = numeric_stats
        
        # Frequency counts for categorical columns (top 5 categories for each)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            categorical_stats = {}
            for col in categorical_columns:
                value_counts = df[col].value_counts().head(5).to_dict()
                categorical_stats[col] = {str(k): int(v) for k, v in value_counts.items()}
                
            result["categorical_summary"] = categorical_stats
        
        # Correlation matrix for numeric columns (if more than one)
        if len(numeric_columns) > 1:
            try:
                corr_matrix = df[numeric_columns].corr().to_dict()
                
                # Convert numpy types to Python native types
                for col1, corrs in corr_matrix.items():
                    for col2, value in corrs.items():
                        corr_matrix[col1][col2] = float(value)
                        
                result["correlation_matrix"] = corr_matrix
            except:
                pass
        
        return result