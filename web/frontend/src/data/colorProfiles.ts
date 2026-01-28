export interface ColorRule {
  maxEquity: number  // 上限百分比，最后一条用 100
  color: string      // 背景色 hex
}

export interface ColorProfile {
  id: string
  name: string
  nameKey: string    // i18n key
  isGradient?: boolean  // 蓝色渐变特殊处理
  rules: ColorRule[]
}

// 预定义配色方案
export const BUILTIN_PROFILES: ColorProfile[] = [
  {
    id: 'gradient',
    name: '蓝色渐变',
    nameKey: 'preflop.profile.gradient',
    isGradient: true,
    rules: [], // 渐变模式不使用 rules
  },
  {
    id: 'traffic-light',
    name: '信号灯',
    nameKey: 'preflop.profile.trafficLight',
    rules: [
      { maxEquity: 20, color: '#F3F4F6' },  // gray-100
      { maxEquity: 25, color: '#DBEAFE' },  // blue-100
      { maxEquity: 33, color: '#DCFCE7' },  // green-100
      { maxEquity: 100, color: '#FEF9C3' }, // yellow-100
    ],
  },
  {
    id: 'heads-up',
    name: '双人桌',
    nameKey: 'preflop.profile.headsUp',
    rules: [
      { maxEquity: 40, color: '#F3F4F6' },  // gray-100
      { maxEquity: 50, color: '#DBEAFE' },  // blue-100
      { maxEquity: 60, color: '#DCFCE7' },  // green-100
      { maxEquity: 100, color: '#FEF9C3' }, // yellow-100
    ],
  },
]

// 默认自定义配置（与信号灯相同）
export const DEFAULT_CUSTOM_PROFILE: ColorProfile = {
  id: 'custom',
  name: '自定义',
  nameKey: 'preflop.profile.custom',
  rules: [
    { maxEquity: 20, color: '#F3F4F6' },
    { maxEquity: 25, color: '#DBEAFE' },
    { maxEquity: 33, color: '#DCFCE7' },
    { maxEquity: 100, color: '#FEF9C3' },
  ],
}

// 根据背景色亮度自动计算文字颜色
export function getTextColorFromBg(bgColor: string): string {
  // 解析 hex 颜色
  const hex = bgColor.replace('#', '')
  const r = parseInt(hex.substring(0, 2), 16)
  const g = parseInt(hex.substring(2, 4), 16)
  const b = parseInt(hex.substring(4, 6), 16)

  // 计算相对亮度 (sRGB)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

  // 亮色背景用深色文字，暗色背景用浅色文字
  return luminance > 0.6 ? '#374151' : '#FFFFFF'
}

// 蓝色渐变计算（保留原有逻辑）
function getGradientColor(equity: number): { bg: string; text: string } {
  const normalized = Math.max(0, Math.min(1, (equity - 5) / 80))

  const hue = 214
  const saturation = 10 + normalized * 70
  const lightness = 96 - normalized * 18

  const bg = `hsl(${hue}, ${saturation}%, ${lightness}%)`
  const text = normalized > 0.6 ? '#1a1a1a' : '#374151'

  return { bg, text }
}

// 根据阈值规则获取颜色
function getRuleColor(equity: number, rules: ColorRule[]): { bg: string; text: string } {
  for (const rule of rules) {
    if (equity < rule.maxEquity) {
      return {
        bg: rule.color,
        text: getTextColorFromBg(rule.color),
      }
    }
  }
  // 默认返回最后一个规则的颜色
  const lastRule = rules[rules.length - 1]
  return {
    bg: lastRule?.color || '#F3F4F6',
    text: getTextColorFromBg(lastRule?.color || '#F3F4F6'),
  }
}

// 获取 profile 对应的颜色
export function getProfileColor(
  equity: number,
  profile: ColorProfile
): { bg: string; text: string } {
  if (profile.isGradient) {
    return getGradientColor(equity)
  }
  return getRuleColor(equity, profile.rules)
}

// localStorage 存储 key
const STORAGE_KEY = 'holdem-lab-custom-color-profile'

// 保存自定义配置到 localStorage
export function saveCustomProfile(profile: ColorProfile): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profile))
  } catch (e) {
    console.error('Failed to save custom color profile:', e)
  }
}

// 从 localStorage 加载自定义配置
export function loadCustomProfile(): ColorProfile | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      return JSON.parse(saved) as ColorProfile
    }
  } catch (e) {
    console.error('Failed to load custom color profile:', e)
  }
  return null
}

// 根据 id 获取 profile
export function getProfileById(id: string, customProfile: ColorProfile | null): ColorProfile {
  if (id === 'custom') {
    return customProfile || DEFAULT_CUSTOM_PROFILE
  }
  return BUILTIN_PROFILES.find(p => p.id === id) || BUILTIN_PROFILES[0]
}
