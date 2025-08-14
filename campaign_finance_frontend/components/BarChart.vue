<template>
  <canvas ref="chart"></canvas>
</template>

<script>
import { ref, onMounted, watch } from 'vue';
import { Chart } from 'chart.js';

export default {
  name: 'BarChart',
  props: {
    data: {
      type: Object,
      required: true,
    },
    options: {
      type: Object,
      default: () => ({
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      }),
    },
  },
  setup(props) {
    const chart = ref(null);
    let chartInstance = null;

    onMounted(() => {
      chartInstance = new Chart(chart.value, {
        type: 'bar',
        data: props.data,
        options: props.options,
      });
    });

    watch(
      () => props.data,
      (newData) => {
        if (chartInstance) {
          chartInstance.data = newData;
          chartInstance.update();
        }
      }
    );

    return {
      chart,
    };
  },
};
</script>
