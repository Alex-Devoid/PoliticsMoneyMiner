<script setup>
import { onMounted, watch, ref } from 'vue'
import * as d3 from 'd3'
import * as topojson from 'topojson-client'

defineProps({
  covered:    { type: Array,     required: true },  // ['ca','nc', …]
  clickState: { type: Function,  required: true }
})

const svgRef = ref(null)

watch(() => covered, draw, { deep: true })
onMounted(draw)

async function draw () {
  if (!svgRef.value) return

  /* 1) load US topojson once (cached on window) */
  if (!window.__usTopo) {
    window.__usTopo = await d3.json(
      'https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json'
    )
  }
  const topo  = window.__usTopo
  const path  = d3.geoPath()

  const svg = d3.select(svgRef.value)
    .attr('viewBox', '0 0 975 610')
    .attr('preserveAspectRatio', 'xMidYMid meet')

  svg.selectAll('*').remove()             // clear for redraw

  svg.append('g')
    .selectAll('path')
    .data(topojson.feature(topo, topo.objects.states).features)
    .join('path')
      .attr('d', path)
      .attr('fill', d =>
        covered.includes(d.properties.iso.toLowerCase()) ? '#4caf50' : '#eee')
      .attr('stroke', '#999')
      .style('cursor', d =>
        covered.includes(d.properties.iso.toLowerCase()) ? 'pointer' : 'default')
      .on('click', (e, d) => {
        const iso = d.properties.iso.toLowerCase()
        if (covered.includes(iso)) clickState(iso.toUpperCase())
      })
}
</script>

<template>
  <svg ref="svgRef" class="w-100 h-auto" />
</template>
