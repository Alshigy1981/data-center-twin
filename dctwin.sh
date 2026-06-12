#!/bin/bash
# DC Twin Control Script
# Usage: ./dctwin.sh start | stop | restart | status | logs

BACKEND_PLIST="$HOME/Library/LaunchAgents/com.dctwin.backend.plist"
DASHBOARD_PLIST="$HOME/Library/LaunchAgents/com.dctwin.dashboard.plist"
LOG_DIR="$HOME/data_center_twin/logs"

case "$1" in
  start)
    echo "Starting DC Twin..."
    launchctl load $BACKEND_PLIST
    launchctl load $DASHBOARD_PLIST
    sleep 3
    echo "Backend:   http://localhost:8000"
    echo "Dashboard: http://localhost:8501"
    echo "API Docs:  http://localhost:8000/docs"
    ;;
  stop)
    echo "Stopping DC Twin..."
    launchctl unload $BACKEND_PLIST
    launchctl unload $DASHBOARD_PLIST
    echo "DC Twin stopped."
    ;;
  restart)
    $0 stop
    sleep 2
    $0 start
    ;;
  status)
    echo "=== DC Twin Status ==="
    if launchctl list | grep -q "com.dctwin.backend"; then
      echo "Backend:   RUNNING ✅"
    else
      echo "Backend:   STOPPED ❌"
    fi
    if launchctl list | grep -q "com.dctwin.dashboard"; then
      echo "Dashboard: RUNNING ✅"
    else
      echo "Dashboard: STOPPED ❌"
    fi
    echo ""
    echo "URLs:"
    echo "  Dashboard: http://localhost:8501"
    echo "  API:       http://localhost:8000"
    echo "  API Docs:  http://localhost:8000/docs"
    ;;
  logs)
    echo "=== Backend Logs (last 50 lines) ==="
    tail -50 $LOG_DIR/backend.log
    echo ""
    echo "=== Dashboard Logs (last 50 lines) ==="
    tail -50 $LOG_DIR/dashboard.log
    ;;
  *)
    echo "Usage: ./dctwin.sh {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
