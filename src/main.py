# src/main.py

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import signal
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from database.database_manager import db_manager
from data.argo_data_ingestor import ArgoDataIngestor
from nlp.query_processor import ArgoQueryProcessor
from visualization.plot_generator import AdvancedArgoPlotGenerator
from config import config

class ArgoAISystem:
    """Main ARGO AI System controller with enhanced features"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.data_ingestor = None
        self.query_processor = None
        self.plot_generator = None
        self.is_running = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        log_level = config.get('logging.level', 'INFO')
        log_format = config.get('logging.format', 
                               '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = config.get('logging.file', 'logs/argo_system.log')
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return logging.getLogger(__name__)
    
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing ARGO AI System...")
            
            # Initialize database
            db_url = config.get('database.url')
            db_echo = config.get('database.echo', False)
            
            if not db_manager.initialize_database(db_url, db_echo):
                self.logger.error("Database initialization failed")
                return False
            
            # Create database tables
            db_manager.create_tables()
            
            # Initialize components
            self.data_ingestor = ArgoDataIngestor()
            self.query_processor = ArgoQueryProcessor()
            self.plot_generator = AdvancedArgoPlotGenerator()
            
            # Warm up NLP model
            self.logger.info("Warming up NLP model...")
            self.query_processor.initialize_model()
            
            self.logger.info("ARGO AI System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def process_data_directory(self, data_dir: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process NetCDF data directory with enhanced monitoring"""
        if not self.data_ingestor:
            raise RuntimeError("System not initialized")
        
        self.logger.info(f"Processing data directory: {data_dir}")
        
        start_time = time.time()
        results = self.data_ingestor.process_netcdf_directory(data_dir, force_reprocess)
        processing_time = time.time() - start_time
        
        # Add system metrics
        results['system_metrics'] = {
            'processing_time_seconds': processing_time,
            'processing_rate_files_per_second': results['processed_files'] / processing_time if processing_time > 0 else 0,
            'memory_usage_mb': self._get_memory_usage(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Data processing completed: {results['successful_files']} files processed")
        return results
    
    def execute_nlp_query(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Execute NLP query with enhanced results"""
        if not self.query_processor:
            raise RuntimeError("System not initialized")
        
        self.logger.info(f"Executing NLP query: '{query}'")
        
        start_time = time.time()
        results = self.query_processor.process_query(query, max_results)
        query_time = time.time() - start_time
        
        # Enhance results with additional information
        results['query_metrics'] = {
            'processing_time_seconds': query_time,
            'query_complexity': len(query.split()),
            'results_count': len(results.get('profiles', [])),
            'cache_hit': results.get('cache_hit', False)
        }
        
        return results
    
    def generate_visualization(self, data: List[Dict], plot_type: str, **kwargs) -> Any:
        """Generate visualization with error handling"""
        if not self.plot_generator:
            raise RuntimeError("System not initialized")
        
        self.logger.info(f"Generating {plot_type} visualization for {len(data)} profiles")
        
        try:
            plot_methods = {
                'map': self.plot_generator.create_interactive_map,
                'profile': self.plot_generator.create_comprehensive_profile_plot,
                'temporal': self.plot_generator.create_temporal_analysis_plot,
                'quality': self.plot_generator.create_quality_dashboard,
                'vertical': self.plot_generator.create_vertical_section_plot,
                'comparison': self.plot_generator.create_comparison_plot,
                'regional': self.plot_generator.create_region_analysis_plot,
                'animated': self.plot_generator.create_animated_timeseries,
                'dashboard': self.plot_generator.create_dashboard
            }
            
            if plot_type not in plot_methods:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            plot = plot_methods[plot_type](data, **kwargs)
            self.logger.info(f"Visualization generated successfully: {plot_type}")
            return plot
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return self.plot_generator._create_empty_plot(f"Error generating plot: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system': {
                'status': 'running' if self.is_running else 'stopped',
                'uptime': getattr(self, '_start_time', None),
                'version': '1.0.0'
            },
            'database': {
                'initialized': db_manager.engine is not None,
                'stats': db_manager.get_database_stats() if db_manager.engine else {}
            },
            'components': {
                'data_ingestor': self.data_ingestor is not None,
                'query_processor': self.query_processor is not None,
                'plot_generator': self.plot_generator is not None
            },
            'resources': {
                'memory_usage_mb': self._get_memory_usage(),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        return status
    
    def run_interactive_mode(self):
        """Run interactive command-line mode"""
        self.is_running = True
        self._start_time = datetime.utcnow()
        
        print("=== ARGO AI System - Interactive Mode ===")
        print("Commands: status, process <dir>, query <text>, plot <type>, exit")
        
        while self.is_running:
            try:
                command = input("\nargo> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'exit':
                    break
                elif cmd == 'status':
                    status = self.get_system_status()
                    print(f"System Status: {status['system']['status']}")
                    print(f"Database: {status['database']['stats']}")
                elif cmd == 'process' and len(command) > 1:
                    results = self.process_data_directory(command[1])
                    print(f"Processed {results['successful_files']} files")
                elif cmd == 'query' and len(command) > 1:
                    query_text = ' '.join(command[1:])
                    results = self.execute_nlp_query(query_text)
                    print(f"Found {len(results.get('profiles', []))} results")
                elif cmd == 'plot' and len(command) > 1:
                    # Simplified plot command for demo
                    print(f"Plot generation for {command[1]} would be displayed")
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown system gracefully"""
        self.logger.info("Shutting down ARGO AI System...")
        self.is_running = False
        
        # Close database connections
        if hasattr(db_manager, 'close_connections'):
            db_manager.close_connections()
        
        self.logger.info("ARGO AI System shutdown complete")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return round(process.memory_info().rss / (1024 * 1024), 2)
        except ImportError:
            return 0.0

def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(description='ARGO AI System')
    parser.add_argument('--data-dir', help='Process NetCDF data directory')
    parser.add_argument('--query', help='Execute NLP query')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--init-db', action='store_true', help='Initialize database only')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config.load_config(args.config)
    
    # Create system instance
    system = ArgoAISystem()
    
    # Initialize system
    if not system.initialize_system():
        print("System initialization failed. Check logs for details.")
        return 1
    
    try:
        if args.init_db:
            print("Database initialized successfully")
            return 0
        
        if args.data_dir:
            results = system.process_data_directory(args.data_dir)
            print(f"Data processing completed: {results}")
            return 0
        
        if args.query:
            results = system.execute_nlp_query(args.query)
            print(f"Query results: {results}")
            return 0
        
        if args.interactive:
            system.run_interactive_mode()
            return 0
        
        # Default: run interactive mode
        system.run_interactive_mode()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        system.shutdown()

if __name__ == "__main__":
    sys.exit(main())