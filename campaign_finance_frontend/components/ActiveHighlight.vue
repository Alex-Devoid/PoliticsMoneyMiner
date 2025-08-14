<script setup>
import { computed } from 'vue'

const props = defineProps({
  bbox     : { type: Object, required: true },  // may be null at runtime
  scale    : { type: Number, required: true },
  active   : { type: Boolean, default: false },
  approved : { type: Boolean, default: false }
})

/* return null style if no bbox ------------------------------------------- */
const styleObj = computed(() => {
  if (!props.bbox) return { display: 'none' }

  const { x, y, w, h } = props.bbox
  const s = props.scale || 1

  return {
    position : 'absolute',
    left     : `${x * s}px`,
    top      : `${y * s}px`,
    width    : `${w * s}px`,
    height   : `${h * s}px`,
    pointerEvents: 'none',

    backgroundColor: props.approved
      ? 'transparent'
      : props.active
        ? 'rgba(49,130,206,0.25)'
        : 'rgba(160,174,192,0.20)',

    border: props.approved
      ? '2px solid #38A169'
      : props.active
        ? '2px solid #3182CE'
        : '1px solid rgba(0,0,0,0.15)',

    borderRadius: '4px',
    boxSizing   : 'border-box'
  }
})
</script>

<template>
  <div :style="styleObj" />
</template>
