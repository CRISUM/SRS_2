// 创建新文件: src/components/TagFilter.js
import React, { useState, useEffect } from 'react';

const TagFilter = ({ availableTags, selectedTags, onTagSelect }) => {
  // 按首字母对标签进行分组
  const [activeGroup, setActiveGroup] = useState('#');

  // 创建标签分组函数
  const groupTagsByFirstLetter = () => {
    const groups = { '#': [] };

    // 为每个字母创建一个空数组
    for (let i = 65; i <= 90; i++) {
      groups[String.fromCharCode(i)] = [];
    }

    // 将标签分配到相应字母组
    availableTags.forEach(tag => {
      if (!tag) return; // 跳过无效标签

      const firstChar = tag.toString().trim().charAt(0).toUpperCase();
      if (/[A-Z]/.test(firstChar)) {
        groups[firstChar].push(tag);
      } else {
        groups['#'].push(tag); // 非字母开头的放在 # 组
      }
    });

    return groups;
  };

  const tagGroups = groupTagsByFirstLetter();

  // 获取非空的分组（有标签的分组）
  const nonEmptyGroups = Object.keys(tagGroups).filter(
    group => tagGroups[group].length > 0
  ).sort();

  // 当可用标签变化时更新活跃组
  useEffect(() => {
    if (nonEmptyGroups.length > 0 && !nonEmptyGroups.includes(activeGroup)) {
      setActiveGroup(nonEmptyGroups[0]);
    }
  }, [availableTags, nonEmptyGroups, activeGroup]);

  // 样式定义
  const containerStyle = {
    backgroundColor: '#f8fafc',
    borderRadius: '0.5rem',
    border: '1px solid #e2e8f0',
    padding: '1rem',
    marginBottom: '1.5rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
  };

  const tabsContainerStyle = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.25rem',
    marginBottom: '1rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid #e2e8f0'
  };

  const tabStyle = (isActive) => ({
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    fontSize: '0.875rem',
    fontWeight: isActive ? '600' : '400',
    cursor: 'pointer',
    backgroundColor: isActive ? '#2563eb' : '#e2e8f0',
    color: isActive ? 'white' : '#4b5563',
    transition: 'all 0.2s'
  });

  const tagsContainerStyle = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
    maxHeight: '200px',
    overflowY: 'auto',
    padding: '0.5rem'
  };

  const tagStyle = (isSelected) => ({
    padding: '0.25rem 0.75rem',
    borderRadius: '9999px',
    fontSize: '0.75rem',
    backgroundColor: isSelected ? '#2563eb' : '#e2e8f0',
    color: isSelected ? 'white' : '#4b5563',
    cursor: 'pointer',
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center'
  });

  return (
    <div style={containerStyle}>
      <h3 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.75rem' }}>
        Filter tags by initials
      </h3>

      {/* 字母选项卡 */}
      <div style={tabsContainerStyle}>
        {nonEmptyGroups.map(group => (
          <div
            key={group}
            style={tabStyle(activeGroup === group)}
            onClick={() => setActiveGroup(group)}
          >
            {group}
          </div>
        ))}
      </div>

      {/* 标签列表区域 */}
      {nonEmptyGroups.length > 0 ? (
        <div style={tagsContainerStyle}>
          {tagGroups[activeGroup].map(tag => (
            <div
              key={tag}
              style={tagStyle(selectedTags.includes(tag))}
              onClick={() => onTagSelect(tag)}
            >
              {tag}
              {selectedTags.includes(tag) && (
                <svg xmlns="http://www.w3.org/2000/svg" className="ml-1 h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: '1rem',
          color: '#6b7280',
          fontSize: '0.875rem'
        }}>
          没有可用的标签
        </div>
      )}

      {/* 已选标签显示区域 */}
      {selectedTags.length > 0 && (
        <div style={{
          marginTop: '1rem',
          paddingTop: '0.75rem',
          borderTop: '1px solid #e2e8f0'
        }}>
          <div style={{
            fontSize: '0.75rem',
            color: '#6b7280',
            marginBottom: '0.5rem'
          }}>
            已选择 {selectedTags.length} 个标签:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
            {selectedTags.map(tag => (
              <div
                key={tag}
                style={{
                  backgroundColor: '#dbeafe',
                  color: '#2563eb',
                  fontSize: '0.75rem',
                  padding: '0.25rem 0.5rem',
                  borderRadius: '9999px',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {tag}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onTagSelect(tag);
                  }}
                  style={{
                    marginLeft: '0.25rem',
                    color: '#3b82f6',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TagFilter;