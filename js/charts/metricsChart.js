/**
 * NeuroForge-3D Metrics Chart
 * Real-time Chart.js telemetry dashboard tracking training loss & accuracy across epochs.
 */

/* global Chart */

export class MetricsChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.chart = null;
    this.maxDataPoints = 60;

    this.epochs = [];
    this.lossData = [];
    this.accData = [];

    this.init();
  }

  init() {
    const ctx = this.canvas.getContext('2d');

    // Gradient fills
    const lossGradient = ctx.createLinearGradient(0, 0, 0, 160);
    lossGradient.addColorStop(0, 'rgba(244, 63, 94, 0.35)');
    lossGradient.addColorStop(1, 'rgba(244, 63, 94, 0.0)');

    const accGradient = ctx.createLinearGradient(0, 0, 0, 160);
    accGradient.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
    accGradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.epochs,
        datasets: [
          {
            label: 'Loss',
            data: this.lossData,
            borderColor: '#f43f5e',
            backgroundColor: lossGradient,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: 0,
            pointHoverRadius: 4,
            yAxisID: 'yLoss'
          },
          {
            label: 'Accuracy',
            data: this.accData,
            borderColor: '#00f0ff',
            backgroundColor: accGradient,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: 0,
            pointHoverRadius: 4,
            yAxisID: 'yAcc'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              boxWidth: 12,
              usePointStyle: true
            }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            borderColor: '#334155',
            borderWidth: 1,
            padding: 8
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.25)' },
            ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 6 }
          },
          yLoss: {
            type: 'linear',
            position: 'left',
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#f43f5e', font: { size: 10 } },
            suggestedMin: 0
          },
          yAcc: {
            type: 'linear',
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: {
              color: '#00f0ff',
              font: { size: 10 },
              callback: val => `${Math.round(val * 100)}%`
            },
            min: 0,
            max: 1
          }
        }
      }
    });
  }

  addPoint(epoch, loss, accuracy) {
    this.epochs.push(epoch);
    this.lossData.push(loss);
    this.accData.push(accuracy);

    if (this.epochs.length > this.maxDataPoints) {
      this.epochs.shift();
      this.lossData.shift();
      this.accData.shift();
    }

    this.chart.update('none'); // high performance zero-animation update
  }

  reset() {
    this.epochs.length = 0;
    this.lossData.length = 0;
    this.accData.length = 0;
    this.chart.update();
  }
}
