from flask import Flask, render_template, jsonify, request
from simulation_engine import engine
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
        stats=engine.get_summary_stats(),
        bounds=engine.get_map_bounds(),
        source=engine.get_source_location())

@app.route('/api/bounds')
def api_bounds():
    return jsonify(engine.get_map_bounds())

@app.route('/api/source')
def api_source():
    return jsonify(engine.get_source_location())

@app.route('/api/routes')
def api_routes():
    return jsonify(engine.get_routes_latlng())

@app.route('/api/sinks')
def api_sinks():
    return jsonify(engine.get_sinks_latlng())

@app.route('/api/rivers')
def api_rivers():
    return jsonify(engine.get_rivers_geojson())

@app.route('/api/analytics')
def api_analytics():
    try:
        data = engine.get_query_analytics()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    d = request.get_json() or {}
    try:
        ts = engine.simulate_flood_wave(
            rainfall_mm            = float(d.get('rainfall_mm', 150)),
            barrage_release_cusecs = float(d.get('barrage_release', 5000)),
            time_hours             = int(d.get('time_hours', 24)),
            scenario               = d.get('scenario', 'normal'),
        )
        return jsonify({'success': True, 'timesteps': ts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 55)
    print("  VIJAYAWADA FLOOD SIMULATION  |  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000, host='0.0.0.0')